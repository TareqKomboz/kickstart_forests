# let's not git track this file, as it contains sensitive information
from secret import MLFLOW_PASSWORD

DATA_BASE_DIR = "data"
MODEL_BASE_DIR = "model"

FOREST_DATA_SET_SEPARATOR = "\t"
FOREST_DATA_SET_INDEX_COL = 0

RANDOM_STATE = 42
NUM_SAMPLES = None

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
USED_FEATURES = COLS_SENTINEL + [COL_WETNESS] + [COL_SPECIES]
TARGET = COL_LAI


MLFLOW_TRACKING_URL = "https://team2-mlflow-3601382-k4avrf3gmq-ey.a.run.app"
MLFLOW_USERNAME = "admin"
MLFLOW_EXPERIMENT = "Forests"

LAKEFS_REPOSITORY = "ai-kickstart"
LAKEFS_BRANCH = "main"