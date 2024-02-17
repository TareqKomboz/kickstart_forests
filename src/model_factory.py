from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from .data import COL_SPECIES, COL_WETNESS, COLS_SENTINEL

class ModelFactory:
    NUMERICAL_COLS = [COL_WETNESS] + COLS_SENTINEL
    CATEGORICAL_COLS = [COL_SPECIES]

    @classmethod
    def create_linear_regression_orig(cls):
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
            ])),
            ('model', LinearRegression())
        ])
    
    @classmethod
    def create_polynomial_regression_orig(cls):
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
            ])),
            ('poly', PolynomialFeatures(degree=2)),
            ('model', LinearRegression())
        ])

    @classmethod
    def create_random_forest_orig(cls):
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
            ])),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    
    @classmethod
    def create_xgboost_orig(cls):
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
            ])),
            ('model', XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.2,
                max_depth = 4, alpha = 3, n_estimators = 100))
        ])
    
    @classmethod
    def create_MLP_orig(cls):
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
            ])),
            ('model', MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=100, alpha=0.001,
                        solver='adam', verbose=10, random_state=42, tol=0.000000001))
        ])
