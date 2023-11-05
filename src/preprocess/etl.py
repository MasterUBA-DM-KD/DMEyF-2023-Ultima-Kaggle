# Third Party Imports
import pandas as pd
from sklearn import datasets


def extract():
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    return X, y


def transform(X: pd.DataFrame, y):
    y = y.map({0: 0, 1: 1, 2: 0})

    print(X.head())
    print(y.head())
    return X, y


def load(X, y):
    return X, y
