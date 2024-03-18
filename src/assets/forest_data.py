"""Forest data set."""

import pandas as pd
# from caseconverter import snakecase
from dagster import asset

from resources.csv_data_set_loader import CSVDataSetLoader
from utils import get_key_prefix


@asset(io_manager_key="csv_io_manager", key_prefix=get_key_prefix())
def forest_data(
    forest_data_set_loader: CSVDataSetLoader,
    num_samples: int = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Forest data set.

    Parameters
    ----------
    forest_data_set_loader : CSVDataSetLoader
        Raw data set loader.

    Returns
    -------
    pd.DataFrame
        Raw Forest data set."""
    df = forest_data_set_loader.load()
    
    # Re-format column names to snake case. The original data set uses several
    # different formats for its column names.
    # raw_data_df = raw_data_df.rename(mapper=snakecase, axis=1)

    if num_samples is not None and num_samples < len(df):
        df = df.sample(num_samples, random_state=random_seed)
        
    return df