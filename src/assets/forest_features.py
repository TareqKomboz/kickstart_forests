"""Ames housing features."""

import pandas as pd
from dagster import asset

from constants import USED_FEATURES, TARGET
from utils import get_key_prefix


@asset(io_manager_key="csv_io_manager", key_prefix=get_key_prefix())
def forest_features(forest_data: pd.DataFrame):
    """Forest features.

    Filter the forest data set for the selected features and target.

    Parameters
    ----------
    forest_data : pd.DataFrame
        Raw Ames housing data set.

    Returns
    -------
    pd.DataFrame
        Data set with selected features and target.
    """
    selected_columns = (
        USED_FEATURES
        + [TARGET]
    )

    return forest_data[selected_columns]
