import pandas as pd
import numpy as np
from typing import List
import os
import torch
from torch.utils.data import Dataset
from Levenshtein import distance as levenshtein
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
tqdm.pandas()
from datetime import datetime
import time

length_seq = 8
dataset = ['ABCDEF', 'BDCE', 'EEEEE', 'EFCD', 'BBAABB']
score = [1, 2, 1, 3, 2]

df_full = pd.DataFrame(np.transpose([dataset, score]),columns=['sequence','score'])

df_full = pd.read_csv('1279050_1_Paired_All.csv', low_memory=False)

for i in range(len(dataset)):
    df_full.loc[i, "sequence"] = df_full.loc[i, "sequence"] + "-"*(length_seq-len(df_full.loc[i, "sequence"]))
    
print(df_full["sequence"])

'''
class SequenceGapToken(Dataset):
    def __init__(
            self,
            *,
            csv_path: str,
            task_dir: str,
            filter_percentile: str,
            min_mutant_dist: int,
            top_quantile: float,
            alphabet: str,
            smoothing_params: str,
            nbhd_params: str,
            smoothed_fname: str = 'results',
            sequence_column: str = 'sequence',
            output_dir: str = '',
            val_samples: int = 0,
            seed: int = 0
        ):
        
        percentile_str = '_'.join([str(x) for x in filter_percentile])
        write_dir = os.path.join(
            task_dir, f'mutations_{min_mutant_dist}', f'percentile_{percentile_str}'
        )
        
        write_path = os.path.join(
            write_dir, f'base_seqs.csv'
        )
        
        os.makedirs(write_dir, exist_ok=True)
        self._write_path = write_path
        self._log = logging.getLogger(__name__)
        self._sequence_column = sequence_column
        self._top_quantile = top_quantile
        self._alphabet = alphabet
        if smoothing_params != 'unsmoothed':
            smoothed_path = os.path.join(write_dir, smoothing_params, nbhd_params, smoothed_fname + '.csv') 
            self._log.info(f'Using smoothed data from {smoothed_path}')
            if not os.path.exists(smoothed_path):
                raise ValueError(f"Could not find smoothed data at {smoothed_path}")
            self._data_df = pd.read_csv(smoothed_path)
                # EOS addition of the gap tokens
                for i in range(len(dataset)):
                    self._data_df.loc[i, "sequence"] = self._data_df.loc[i, "sequence"] + "-"*(length_seq-len(self._data_df.loc[i, "sequence"]))
            self._log.info(f'Read in {len(self._data_df)} smoothed sequences.')
        else:
            self._log.info(f"Reading CSV file {csv_path}")
            _raw_data_df = pd.read_csv(csv_path)
            prev_num_rows = _raw_data_df.shape[0]
            self._data_df = self._filter(_raw_data_df, filter_percentile, min_mutant_dist)
            new_num_rows = self._data_df.shape[0]
            self._log.info(
                f"Filtered {prev_num_rows} to {new_num_rows} rows in {filter_percentile} "
                + f"score range and >={min_mutant_dist} mutations away.")
            self._data_df.to_csv(self._write_path, index=False)
            
        self._seq_len = len(self._data_df[sequence_column].iloc[0])
        self._log.info(f"Dataset has {len(self._data_df)} variants")

    def _filter(self, data_df, percentile, min_mutant_dist):
        lower_value = data_df.score.quantile(percentile[0])
        upper_value = data_df.score.quantile(percentile[1])
        top_quantile = data_df.score.quantile(self._top_quantile)
        top_sequences_df = data_df[data_df.score >= top_quantile]  
        
        self._log.info('Filtering')
        filtered_df = data_df[data_df.score.between(lower_value, upper_value)]
        if min_mutant_dist == 0:
            return filtered_df
        get_min_dist = lambda x: np.min([levenshtein(x.strip(), top_seq.strip()) for top_seq in top_sequences_df.sequence]) 
        self._log.info('Getting minimum Levenshtein distance to top sequences')
        mutant_dist = filtered_df.sequence.progress_map(get_min_dist)
        return filtered_df[mutant_dist >= min_mutant_dist].reset_index(drop=True)
        
    # @property
    # def sequences(self):
        # return self._data.sequence.tolist()
'''