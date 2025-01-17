"""Train and test data sets."""

from typing import Tuple

import pandas as pd
from dagster import AssetOut, multi_asset
from sklearn.model_selection import train_test_split

from constants import RANDOM_STATE
from utils import get_key_prefix


@multi_asset(
    outs={
        "train_data": AssetOut(
            io_manager_key="csv_io_manager", key_prefix=get_key_prefix()
        ),
        "test_data": AssetOut(
            io_manager_key="csv_io_manager", key_prefix=get_key_prefix()
        ),
    }
)
def train_test_data(
    forest_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data set into a train and test data set.

    Parameters
    ----------
    ames_housing_features : pd.DataFrame
        Ames housing data features and target

    Returns
    -------
    pd.DataFrame
        Training data set
    pd.DataFrame
        Test data set
    """
    train_data: pd.DataFrame
    test_data: pd.DataFrame

    train_data, test_data = train_test_split(
        forest_features,
        random_state=RANDOM_STATE,
    )  # type: ignore
    return train_data, test_data
