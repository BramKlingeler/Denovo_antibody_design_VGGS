from typing import Any, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, random_split, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
from ggs.data.sequence_dataset import SequenceDataset
import numpy as np
import logging
import random
import torch

class PandasDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sequence = self.dataset.iloc[index]['sequence']
        target = self.dataset.iloc[index]['target']
        return sequence, target

    def __len__(self):
        return len(self.dataset)

class PredictorDataModule(LightningDataModule):

    def __init__(
            self,
            *,
            task: str,
            task_cfg: Dict[str, Any],
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            encoding: str, # how to prepare the fasta file for the model
            alphabet: str, # amino acid alphabet
            val_samples: float,
            seed: int,
            sequence_column: str,
            weighted_sampling: bool,
            _preloaded_dataset=None
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        # Data paths
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._encoding = encoding
        self._seed = seed if seed is not None else 42
        self._weighted_sampling = weighted_sampling
        if _preloaded_dataset is not None:
            self._dataset = _preloaded_dataset 
        elif task in {'GFP', 'COV', 'AAV'}: #added 'COV' dataset
            self._dataset = SequenceDataset(
                **task_cfg,
                alphabet=alphabet,
                seed=self._seed,
                sequence_column=sequence_column,
                val_samples=val_samples,
            )
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.cross_validation = False
        self.num_folds = 5
        self.fold_index = 0
        self._log.info(f'Dataset: {len(self._dataset)} examples from the screen')
        

    def setup(self, stage=None):
        if hasattr(self, "_dataset_split_done") and self._dataset_split_done:
            return

        if getattr(self, "cross_validation", False):
            '''
            # Only create subsets; reuse preloaded dataset
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self._seed)
            indices = np.arange(len(self._dataset))
            train_idx, test_idx = list(kf.split(indices))[self.fold_index]

            # Optional: small validation split per fold
            val_size = int(0.1 * len(train_idx))
            train_size = len(train_idx) - val_size
            train_idx, val_idx = train_idx[:train_size], train_idx[train_size:]

            self.train_dataset = Subset(self._dataset, train_idx)
            self.val_dataset = Subset(self._dataset, val_idx)
            self.test_dataset = Subset(self._dataset, test_idx)
            '''
        else:
            n = len(self._dataset)
            n_train = int(0.8 * n) #default 0.8
            n_val = int(0.1 * n) #default 0.1
            n_test = n - n_train - n_val
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self._dataset, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self._seed)
            )
            
        self._dataset_split_done = True

    
    def train_dataloader(self):
        sampler = None
        if self._weighted_sampling:
            '''
            If we are performing weighted sampling, we assume the weight of an example are inversely proportional value of that example's score
            Target values can be negative, so we add the minimum score value to all scores to make them positive
            '''
            self._log.info('Using weighted sampling')
            targets = self.train_dataset.dataset._data_df.score.iloc[self.train_dataset.indices]
            adjusted_targets = targets - targets.min() + 1
            weights = 1 / adjusted_targets.values
            sampler = WeightedRandomSampler(weights, len(weights))
            # torch_dataset = data.DataFrame(x=self._dataset._data_df.drop('target',axis=1),
            #                                y=self._dataset._data_df.score)  
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            sampler=sampler,
            shuffle=(sampler is None)
        )
    

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False
        )
