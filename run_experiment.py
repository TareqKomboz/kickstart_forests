
from sklearn.model_selection import train_test_split
from src.data import Dataset, COL_LAI, COL_SPECIES, COL_WETNESS, COLS_SENTINEL
from src.model_factory import ModelFactory
from src.features import FeaturePipelineFactory
import numpy as np
from logging import getLogger, StreamHandler, INFO, FileHandler
import logging


logger = getLogger()
logger.addHandler(FileHandler("experiment.log"))
logger.setLevel(INFO)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.handlers[0].setFormatter(f_format)


def main():    
    # load data
    dataset = Dataset(10000)

    X, y = dataset.load_xy()

    # split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # val set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    models = [
        ModelFactory.create_linear_regression_orig(feature_set=FeaturePipelineFactory.ALL_FEATURES),
        ModelFactory.create_polynomial_regression_orig(feature_set=FeaturePipelineFactory.ALL_FEATURES),
        ModelFactory.create_random_forest_orig(feature_set=FeaturePipelineFactory.ALL_FEATURES),
        ModelFactory.create_xgboost_orig(feature_set=FeaturePipelineFactory.ALL_FEATURES),
        ModelFactory.create_MLP_orig(feature_set=FeaturePipelineFactory.ALL_FEATURES),

        ModelFactory.create_random_forest_orig(feature_set=FeaturePipelineFactory.SENTINEL_FEATURES),
        ModelFactory.create_random_forest_orig(feature_set=FeaturePipelineFactory.SPECIES_SENTINEL_FEATURES),
        ModelFactory.create_random_forest_orig(feature_set=FeaturePipelineFactory.WETNESS_SENTINEL_FEATURES),
        ModelFactory.create_random_forest_orig(feature_set=FeaturePipelineFactory.ALL_EXP_FEATURES),
    ]

    logger.info("Experiment started...")
    logger.info(f"Number of models: {len(models)}")

    # evaluate models
    for model in models:
        logger.info(f"Model: {model.named_steps['model'].__class__.__name__}, Feature set: {model.named_steps['preprocess'].__class__.__name__}")

        model.fit(X_train, y_train)
        
        logger.info(f"Model score: {model.score(X_test, y_test)}")

if __name__ == '__main__':
    main()