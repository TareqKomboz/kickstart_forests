from .config import csv_data_path
import pandas as pd
import numpy as np
from typing import Optional, Tuple

COL_LAI = 'lai'
COL_SPECIES = 'treeSpecies'
COL_WETNESS = 'wetness'
COLS_SENTINEL = [
    "Sentinel_2A_492.4",
    "Sentinel_2A_559.8",
    "Sentinel_2A_664.6",
    "Sentinel_2A_704.1",
    "Sentinel_2A_740.5",
    "Sentinel_2A_782.8",
    "Sentinel_2A_832.8",
    "Sentinel_2A_864.7",
    "Sentinel_2A_1613.7",
    "Sentinel_2A_2202.4"
]



class Dataset:
    def __init__(self, num_samples: Optional[int] = None, random_seed: int = 42):
        self.num_samples = num_samples
        self.random_seed = random_seed

        # self.y = self.df['lai']
        # self.X = self.df.iloc[:,1:13]
        # self.X_sentinel = self.df.iloc[:,3:13]
        # self.X_species_sentinel = self.df.iloc[:,2:13]
        # self.X_wetness_sentinel = self.df.iloc[:,[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]


    def load_df(self) -> pd.DataFrame:
        csv_path = csv_data_path()
        df = pd.read_csv(csv_path, index_col= 0)
        if self.num_samples is not None and self.num_samples < len(df):
            df = df.sample(self.num_samples, random_state=self.random_seed)
        return df
    
    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.load_df()
        y = df['lai']
        X = df[COLS_SENTINEL+ [COL_WETNESS]+ [COL_SPECIES]]
        return X, y