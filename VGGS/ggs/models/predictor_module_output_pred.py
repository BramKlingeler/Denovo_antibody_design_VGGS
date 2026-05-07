from typing import Any, List, Tuple
import os

import torch
import pandas as pd
import pytorch_lightning as L
from torchmetrics import (
    MinMetric,
    MaxMetric,
    MeanMetric,
    SpearmanCorrCoef,
    PearsonCorrCoef,
    MeanAbsoluteError,
)

from ggs.models.predictors import BaseCNN


class PredictorModule(L.LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    """
    
    
    def __init__(self, model_cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg"])

        self._cfg = model_cfg
        self.min_fluorescence = 0.0

        # --------------------
        # Model + optimizer
        # --------------------
        self.predictor = BaseCNN(**self._cfg.predictor)
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            **self._cfg.optimizer,
        )

        # --------------------
        # Loss
        # --------------------
        self.criterion = torch.nn.MSELoss()

        # --------------------
        # Metrics
        # --------------------
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_sr = SpearmanCorrCoef()
        self.val_sr = SpearmanCorrCoef()
        self.test_sr = SpearmanCorrCoef()

        self.train_pr = PearsonCorrCoef()
        self.val_pr = PearsonCorrCoef()
        self.test_pr = PearsonCorrCoef()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.val_sr_best = MaxMetric()
        self.val_pr_best = MaxMetric()
        self.val_mae_best = MinMetric()

        # --------------------
        # Prediction storage
        # --------------------
        self._max_saved_preds = getattr(self._cfg, "max_saved_predictions", 100_000)
        self._collect_preds = False

        self._train_preds: List[Tuple[str, float, float]] = []
        self._val_preds: List[Tuple[str, float, float]] = []
        self._test_preds: List[Tuple[str, float, float]] = []

    # --------------------------------------------------------------------- #
    # Core
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)

    def model_step(self, batch: Any):
        xs, targets = batch
        targets = targets.float()
        preds = self(xs)
        loss = self.criterion(preds, targets)
        return loss, preds, targets, xs

    # --------------------------------------------------------------------- #
    # Training
    # --------------------------------------------------------------------- #

    def on_train_epoch_start(self):
        # Only collect predictions on LAST epoch
        if self.current_epoch == self.trainer.max_epochs - 1:
            self._collect_preds = True
            self._train_preds.clear()
            self._val_preds.clear()
            self._test_preds.clear()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, xs = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)

        mask = targets > self.min_fluorescence
        if mask.any():
            p, t = preds[mask], targets[mask]
            self.train_sr(p, t)
            self.train_pr(p, t)
            self.train_mae(p, t)

        self.log_dict(
            {
                "train/spearmanr": self.train_sr,
                "train/pearsonr": self.train_pr,
                "train/mae": self.train_mae,
            },
            on_epoch=True,
            prog_bar=True,
        )

        if self._collect_preds:
            self._store_predictions(self._train_preds, xs, preds, targets)

        return loss

    # --------------------------------------------------------------------- #
    # Validation
    # --------------------------------------------------------------------- #

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, xs = self.model_step(batch)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)

        mask = targets > self.min_fluorescence
        if mask.any():
            p, t = preds[mask], targets[mask]
            self.val_sr(p, t)
            self.val_pr(p, t)
            self.val_mae(p, t)

        self.log_dict(
            {
                "val/spearmanr": self.val_sr,
                "val/pearsonr": self.val_pr,
                "val/mae": self.val_mae,
            },
            on_epoch=True,
            prog_bar=True,
        )

        if self._collect_preds:
            self._store_predictions(self._val_preds, xs, preds, targets)

    def on_validation_epoch_end(self):
        self.val_sr_best(self.val_sr.compute())
        self.val_pr_best(self.val_pr.compute())
        self.val_mae_best(self.val_mae.compute())

        self.log_dict(
            {
                "val/spearmanr_best": self.val_sr_best.compute(),
                "val/pearsonr_best": self.val_pr_best.compute(),
                "val/mae_best": self.val_mae_best.compute(),
            },
            prog_bar=True,
        )

    # --------------------------------------------------------------------- #
    # Test
    # --------------------------------------------------------------------- #

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, xs = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)

        mask = targets > self.min_fluorescence
        if mask.any():
            p, t = preds[mask], targets[mask]
            self.test_sr(p, t)
            self.test_pr(p, t)
            self.test_mae(p, t)

        self.log_dict(
            {
                "test/spearmanr": self.test_sr,
                "test/pearsonr": self.test_pr,
                "test/mae": self.test_mae,
            },
            on_epoch=True,
            prog_bar=True,
        )

        self._store_predictions(self._test_preds, xs, preds, targets)

    # --------------------------------------------------------------------- #
    # Prediction storage
    # --------------------------------------------------------------------- #

    def _store_predictions(self, buffer, xs, preds, targets):
        if len(buffer) >= self._max_saved_preds:
            return

        for i in range(len(preds)):
            if len(buffer) >= self._max_saved_preds:
                break

            buffer.append(
                (
                    self._to_sequence(xs[i]),
                    float(preds[i].detach().cpu()),
                    float(targets[i].detach().cpu()),
                )
            )

    def _to_sequence(self, x):
        """
        Override this if xs are tokenized tensors.
        """
        if isinstance(x, str):
            return x
        return str(x.detach().cpu().numpy())

    # --------------------------------------------------------------------- #
    # Write CSVs
    # --------------------------------------------------------------------- #

    def _write_predictions(self, records, split: str):
        if not records:
            return

        df = pd.DataFrame(records, columns=["sequence", "prediction", "target"])
        df["error"] = df["prediction"] - df["target"]

        out_dir = self.trainer.log_dir or self.trainer.default_root_dir
        path = os.path.join(out_dir, f"{split}_predictions.csv")
        df.to_csv(path, index=False)

    def on_train_end(self):
        self._write_predictions(self._train_preds, "train")
        self._write_predictions(self._val_preds, "val")

    def on_test_end(self):
        self._write_predictions(self._test_preds, "test")

    # --------------------------------------------------------------------- #
    # Optimizer
    # --------------------------------------------------------------------- #

    def configure_optimizers(self):
        return self.optimizer
