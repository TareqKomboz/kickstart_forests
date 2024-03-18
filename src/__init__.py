"""Ames Housing price prediction model."""

import os

from dagster import Definitions

from assets.forest_data import forest_data
from assets.forest_features import forest_features
from assets.lai_prediction_models import (
    price_prediction_gradient_boosting_model,
    price_prediction_linear_regression_model,
    price_prediction_random_forest_model,
)
from assets.train_test import train_test_data
from constants import (
    # FOREST_DATA_SET_SEPARATOR,
    # AMES_HOUSING_DATA_SET_URL,
    DATA_BASE_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_PASSWORD,
    MLFLOW_TRACKING_URL,
    MLFLOW_USERNAME,
    MODEL_BASE_DIR,
    # NUM_SAMPLES,
    # RANDOM_STATE,
    FOREST_DATA_SET_SEPARATOR,
    FOREST_DATA_SET_INDEX_COL,
)
from io_managers.csv_fs_io_manager import CSVFileSystemIOManager
from io_managers.csv_lakefs_io_manager import CSVLakeFSIOManager
from io_managers.pickle_fs_io_manager import PickleFileSystemIOManager
from io_managers.pickle_lakefs_io_manager import PickleLakeFSIOManager
from resources.csv_data_set_loader import CSVDataSetLoader
from resources.mlflow_session import MlflowSession

# Depending on the environment, serialize assets to the local file system or to lakeFS.
if os.environ.get("ENV") == "production":
    csv_io_manager = CSVLakeFSIOManager()
    pickle_io_manager = PickleLakeFSIOManager()
else:
    csv_io_manager = CSVFileSystemIOManager(base_dir=DATA_BASE_DIR)
    pickle_io_manager = PickleFileSystemIOManager(base_dir=MODEL_BASE_DIR)


definitions = Definitions(
    assets=[
        forest_data,
        forest_features,
        train_test_data,
        price_prediction_linear_regression_model,
        price_prediction_random_forest_model,
        price_prediction_gradient_boosting_model,
    ],
    resources={
        "ames_housing_data_set_loader": CSVDataSetLoader(
            index_col=FOREST_DATA_SET_INDEX_COL,
            separator=FOREST_DATA_SET_SEPARATOR,
        ),
        "mlflow_session": MlflowSession(
            tracking_url=MLFLOW_TRACKING_URL,
            username=MLFLOW_USERNAME,
            password=MLFLOW_PASSWORD,
            experiment=MLFLOW_EXPERIMENT,
        ),
        "csv_io_manager": csv_io_manager,
        "pickle_io_manager": pickle_io_manager,
        "csv_lakefs_io_manager": CSVLakeFSIOManager(),
    },
)