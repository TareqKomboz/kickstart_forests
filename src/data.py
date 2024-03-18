from .config import csv_data_path
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from constants import USED_FEATURES, TARGET



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
        y = df[TARGET]
        X = df[USED_FEATURES]
        return X, y