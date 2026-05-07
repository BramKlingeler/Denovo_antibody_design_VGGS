import os
import numpy as np
from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as L
import pyrootutils
from datetime import datetime
import torch
from torch.utils.data import Subset, random_split
from sklearn.model_selection import KFold
from statistics import mean, stdev
import copy
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from omegaconf import DictConfig
from omegaconf import OmegaConf
import wandb

from ggs.models.predictors import BaseCNN
from ggs.data.utils.tokenize import Encoder
from ggs.data.predictor_data_module import PredictorDataModule
from ggs.models.predictor_module import PredictorModule
from pytorch_lightning.trainer import Trainer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from ggs import utils

log = utils.get_pylogger(__name__)


def train(cfg: DictConfig) -> Tuple[dict, dict]:

    log.info("Starting standard train/val/test training.")
    
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)   

    # Set-up data
    if cfg.data.task == 'GFP':
        task_cfg = cfg.experiment.gfp
    elif cfg.data.task == 'COV':
        task_cfg = cfg.experiment.cov
    elif cfg.data.task == 'AAV':
        task_cfg = cfg.experiment.aav
    else:
        raise ValueError(f"Unknown task: {cfg.data.task}")
    filter_range = task_cfg.filter_percentile
    log.info(f'Training predictor on task {cfg.data.task}')
    datamodule: LightningDataModule = PredictorDataModule(
        **cfg.data,
        task_cfg=task_cfg,
    )
    
    write_path = datamodule._dataset._write_path
    log.info(
        f"Preprocessed base sequences has saved to {write_path}.")

    if cfg.debug or not cfg.log:
        logger = None
        log.info("Not logging to wandb...")
    else:
        log.info("Instantiating loggers...")
        if cfg.wandb.name is None:
            wandb_name = (
                'range_'
                + '_'.join([str(x) for x in filter_range])
                + '_mutations_' + str(task_cfg.min_mutant_dist)
            )
        else:
            wandb_name = cfg.wandb.name
        wandb.init(project=cfg.wandb.project, name=wandb_name, tags=cfg.tags, mode = 'offline')
        logger = WandbLogger(**cfg.wandb)
    # Set-up model
    model: LightningModule = PredictorModule(cfg.model)

    callbacks_cfg = cfg.callbacks
    percentile = '_'.join([str(x) for x in filter_range])
    
    smoothing_params = task_cfg.smoothing_params
    nbhd_params = task_cfg.nbhd_params if task_cfg.nbhd_params else ''
    output_dir = task_cfg.output_dir if task_cfg.output_dir else datetime.now().strftime("%m_%d_%Y_%H_%M") 
    
    ckpt_dir = os.path.join(
        callbacks_cfg.model_checkpoint.dirpath,
        f'mutations_{task_cfg.min_mutant_dist}',
        f'percentile_{percentile}',
        f'{smoothing_params}_smoothed',
        f'{nbhd_params}',
        f'{output_dir}'
    )
    
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks_cfg.model_checkpoint.dirpath = ckpt_dir
    log.info(f'Model checkpoints being saved to: {ckpt_dir}')
    callbacks: List[Callback] = utils.instantiate_callbacks(callbacks_cfg)
    trainer: Trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger, check_val_every_n_epoch=10, devices=[torch.cuda.current_device()]) # requires GPU
    cfg.model.predictor.seq_len = datamodule._dataset._seq_len

    # Write config to same directory as checkpoints
    cfg_path = os.path.join(ckpt_dir, 'config.yaml')
    with open(cfg_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

  

def run_cross_validation(cfg: DictConfig, num_folds: int = 5):
    
    log.info(f"Starting {num_folds}-fold cross-validation")

    # Set seeds consistently
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Select task configuration
    if cfg.data.task == 'GFP':
        task_cfg = cfg.experiment.gfp
    elif cfg.data.task == 'COV':
        task_cfg = cfg.experiment.cov
    elif cfg.data.task == 'AAV':
        task_cfg = cfg.experiment.aav
    else:
        raise ValueError(f"Unknown task: {cfg.data.task}")


    if cfg.debug or not cfg.log:
        logger = None
        log.info("Not logging to wandb...")
    else:
        log.info("Instantiating loggers...")
        if cfg.wandb.name is None:
            wandb_name = (
                'range_'
                + '_'.join([str(x) for x in filter_range])
                + '_mutations_' + str(task_cfg.min_mutant_dist)
            )
        else:
            wandb_name = cfg.wandb.name
        wandb.init(project=cfg.wandb.project, name=wandb_name, tags=cfg.tags, mode = 'offline')
        logger = WandbLogger(**cfg.wandb)
        
    
    # Create dataset ONCE (full dataset before split)
    full_datamodule = PredictorDataModule(
        **cfg.data,
        task_cfg=task_cfg,
    )
    full_dataset = full_datamodule._dataset

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=cfg.seed)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(full_dataset)):
        log.info(f"\n===== Fold {fold_idx+1}/{num_folds} =====")
        

        # Split manually
        train_subset = Subset(full_dataset, train_idx)       
        test_subset = Subset(full_dataset, test_idx)

        # Sample 50% of test as validation
        gen_seed = int(cfg.seed) if cfg.seed is not None else 42
        val_size = int(0.5 * len(test_subset))
        test_size = len(test_subset) - val_size
        test_data, val_data = random_split(
            test_subset,
            [test_size, val_size],
            generator=torch.Generator().manual_seed(gen_seed)
        )
      
        log.info(
            f"  Train: {len(train_subset)} samples "
            f"({len(train_subset)/len(full_dataset)*100:.2f}%)\n"
            f"  Val:   {len(val_data)} samples "
            f"({len(val_data)/len(full_dataset)*100:.2f}%)\n"
            f"  Test:  {len(test_data)} samples "
            f"({len(test_data)/len(full_dataset)*100:.2f}%)"
)

        # Create a new DataModule wrapper each fold
        dm = PredictorDataModule(
            **cfg.data,
            task_cfg=task_cfg,
            _preloaded_dataset=full_dataset,
        )
        dm.train_dataset = train_subset
        dm.val_dataset = val_data
        dm.test_dataset = test_data
        dm._batch_size = cfg.data.batch_size
        dm._num_workers = cfg.data.num_workers
        dm._pin_memory = cfg.data.pin_memory

        # Model per fold
        model: LightningModule = PredictorModule(cfg.model)

        # Trainer & callbacks
        callbacks = utils.instantiate_callbacks(cfg.callbacks)
        trainer = Trainer(
            **cfg.trainer,
            callbacks=callbacks, 
            logger=logger,
            check_val_every_n_epoch=10,
            devices=[torch.cuda.current_device()]
        )

        log.info("Training fold...")
        trainer.fit(model, dm)
        log.info("Testing fold...")
        metrics = trainer.test(model, dm)[0]
        results.append(metrics)

    # ---- Aggregate metrics ----
    aggregated_results = {}
    aggregated_std = {}

    for k in results[0].keys():
        values = [res[k] for res in results]
        aggregated_results[k] = float(np.mean(values))
        aggregated_std[k] = float(np.std(values))

    log.info("\n===== Cross-Validation Results =====")
    for metric in aggregated_results.keys():
        mean_val = aggregated_results[metric]
        std_val = aggregated_std[metric]
        log.info(f"{metric}: mean = {mean_val:.4f}, std = {std_val:.4f}")

    # Optionally return both mean and std
    return {
        "mean": aggregated_results,
        "std": aggregated_std
    }



@hydra.main(version_base="1.3", config_path="../configs", config_name="train_predictor.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    
    # train the model
    if cfg.get("cross_validation", False):
        num_folds = cfg.get("num_folds", 10)
        run_cross_validation(cfg, num_folds)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
