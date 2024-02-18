from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from .data import COL_SPECIES, COL_WETNESS, COLS_SENTINEL

class ExponentiateTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for exponentiating numerical features."""
    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        return np.exp(X)  # Exponentiate each feature


class FeaturePipelineFactory:
    ALL_FEATURES = 'all_features'
    SENTINEL_FEATURES = 'sentinel_features'
    SPECIES_SENTINEL_FEATURES = 'species_sentinel_features'
    WETNESS_SENTINEL_FEATURES = 'wetness_sentinel_features'
    ALL_EXP_FEATURES = 'all_exp_features'


    NUMERICAL_COLS = [COL_WETNESS] + COLS_SENTINEL
    CATEGORICAL_COLS = [COL_SPECIES]

    @classmethod
    def get_feature_pipeline(cls, feature_set):
        if feature_set is None or feature_set == cls.ALL_FEATURES:
            return cls.get_all_features()
        elif feature_set == cls.SENTINEL_FEATURES:
            return cls.get_sentinel_features()
        elif feature_set == cls.SPECIES_SENTINEL_FEATURES:
            return cls.get_species_sentinel_features()
        elif feature_set == cls.WETNESS_SENTINEL_FEATURES:
            return cls.get_wetness_sentinel_features()
        elif feature_set == cls.ALL_EXP_FEATURES:
            return cls.get_all_exp_features()
        else:
            raise ValueError(f"Invalid feature set: {feature_set}")

    @classmethod
    def get_all_features(cls):
        return Pipeline([
            ('preprocess', ColumnTransformer([
                ('impute_scale_num', Pipeline([
                    ('impute_num', SimpleImputer(strategy='mean')),
                    ('scale', StandardScaler())
                ]), cls.NUMERICAL_COLS),
                ('impute_encode_cat', Pipeline([
                    ('impute_cat', SimpleImputer(strategy="most_frequent")),
                    ('encode_cat', OneHotEncoder(handle_unknown='ignore'))
                ]), cls.CATEGORICAL_COLS)
            ]))])
    
    @classmethod
    def get_sentinel_features(cls):
        return Pipeline([
            ('preprocess', ColumnTransformer([
                ('impute_scale_num', Pipeline([
                    ('impute_num', SimpleImputer(strategy='mean')),
                    ('scale', StandardScaler())
                ]), COLS_SENTINEL)
            ]))])
    
    @classmethod
    def get_species_sentinel_features(cls):
        return Pipeline([
            ('preprocess', ColumnTransformer([
                ('impute_scale_num', Pipeline([
                    ('impute_num', SimpleImputer(strategy='mean')),
                    ('scale', StandardScaler())
                ]), COLS_SENTINEL),
                ('impute_encode_cat', Pipeline([
                    ('impute_cat', SimpleImputer(strategy="most_frequent")),
                    ('encode_cat', OneHotEncoder(handle_unknown='ignore'))
                ]), cls.CATEGORICAL_COLS)
            ]))])
    
    @classmethod
    def get_wetness_sentinel_features(cls):
        return Pipeline([
            ('preprocess', ColumnTransformer([
                ('impute_scale_num', Pipeline([
                    ('impute_num', SimpleImputer(strategy='mean')),
                    ('scale', StandardScaler())
                ]), cls.NUMERICAL_COLS)
            ]))])
    
    @classmethod
    def get_all_exp_features(cls):
        # Pipeline for numerical features: impute, scale, and exponentiate
        numerical_pipeline = Pipeline([
            ('impute_scale_num', Pipeline([
                ('impute_num', SimpleImputer(strategy='mean')),
                ('scale', StandardScaler())
            ])),
            ('exp', ExponentiateTransformer())  # Apply exponentiation
        ])

        # Combined pipeline for numerical features: original and exponentiated
        combined_num_pipeline = FeatureUnion([
            ('original', Pipeline([
                ('impute_num', SimpleImputer(strategy='mean')),
                ('scale', StandardScaler())
            ])),
            ('exp', numerical_pipeline)
        ])

        # Pipeline for categorical features: impute and encode
        categorical_pipeline = Pipeline([
            ('impute_cat', SimpleImputer(strategy="most_frequent")),
            ('encode_cat', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Full preprocessing pipeline
        return Pipeline([
            ('preprocess', ColumnTransformer([
                ('num', combined_num_pipeline, cls.NUMERICAL_COLS),
                ('cat', categorical_pipeline, cls.CATEGORICAL_COLS)
                ]))
            ])
