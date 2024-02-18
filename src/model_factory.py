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

from .features import FeaturePipelineFactory

class ModelFactory:

    @classmethod
    def create_linear_regression_orig(cls, feature_set=None):
        if feature_set is None:
            feature_set = FeaturePipelineFactory.ALL_FEATURES
        return Pipeline([
            ('preprocess', FeaturePipelineFactory.get_feature_pipeline(feature_set)),
            ('model', LinearRegression())
        ])
    
    @classmethod
    def create_polynomial_regression_orig(cls, feature_set=None):
        if feature_set is None:
            feature_set = FeaturePipelineFactory.ALL_FEATURES
        
        return Pipeline([
            ('preprocess', FeaturePipelineFactory.get_feature_pipeline(feature_set)),
            ('poly', PolynomialFeatures(degree=2)),
            ('model', LinearRegression())
        ])

    @classmethod
    def create_random_forest_orig(cls, feature_set=None):
        if feature_set is None:
            feature_set = FeaturePipelineFactory.ALL_FEATURES
        return Pipeline([
            ('preprocess', FeaturePipelineFactory.get_feature_pipeline(feature_set)),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    
    @classmethod
    def create_xgboost_orig(cls, feature_set=None):
        if feature_set is None:
            feature_set = FeaturePipelineFactory.ALL_FEATURES
        return Pipeline([
            ('preprocess', FeaturePipelineFactory.get_feature_pipeline(feature_set)),
            ('model', XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.2,
                max_depth = 4, alpha = 3, n_estimators = 100, random_state=42))
        ])
    
    @classmethod
    def create_MLP_orig(cls, feature_set=None):
        if feature_set is None:
            feature_set = FeaturePipelineFactory.ALL_FEATURES
        return Pipeline([
            ('preprocess', FeaturePipelineFactory.get_feature_pipeline(feature_set)),
            ('model', MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=100, alpha=0.001,
                        solver='adam', verbose=0, random_state=42, tol=0.000000001))
        ])
