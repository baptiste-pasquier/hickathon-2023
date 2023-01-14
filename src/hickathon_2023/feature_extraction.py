import pandas as pd
from sklearn.base import BaseEstimator


def compute_year(df):
    df["year"] = pd.to_datetime(df["consumption_measurement_date"]).dt.year
    return df


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        X = compute_year(X)
        return X
