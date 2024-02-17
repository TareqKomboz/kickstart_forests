
from sklearn.model_selection import train_test_split
from src.data import Dataset, COL_LAI, COL_SPECIES, COL_WETNESS, COLS_SENTINEL
from src.model_factory import ModelFactory
import numpy as np

def main():    
    # load data
    dataset = Dataset(10000)

    X, y = dataset.load_xy()

    # split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # val set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    # define models to be evaluated
    # these models are evaluated on all 12 features, we later want to refactor to have feature registry,
    # so that we can evaluate models on different feature sets and on differnet feature engineering such as 
    # exponential transformation of satellite data, or concatenation of species and satellite data
    models = [
        ModelFactory.create_linear_regression_orig(),
        ModelFactory.create_polynomial_regression_orig(),
        ModelFactory.create_random_forest_orig(),
        ModelFactory.create_xgboost_orig(),
        ModelFactory.create_MLP_orig(),   
    ]

    # evaluate models
    for model in models:
        print(f"Evaluating model:\n{model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Model score: {model.score(X_test, y_test)}")
        print(f"Model mse: {np.mean((y_pred - y_test)**2)}")    


if __name__ == '__main__':
    main()