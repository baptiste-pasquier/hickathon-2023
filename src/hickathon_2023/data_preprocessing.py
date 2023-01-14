import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from hickathon_2023.utils import Pipeline


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        return X[self.features]


def preprocess_thermal_intertia(df):
    dic_inertia = {"low": 1, "medium": 2, "high": 3, "very high": 4}
    df["thermal_inertia"] = df["thermal_inertia"].replace(dic_inertia)
    return df


class DataProcessing(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        X = preprocess_thermal_intertia(X)
        return X


def get_data_preprocessor():

    preprocess = DataProcessing()
    selector = FeatureSelector(
        features=["year", "living_area_sqft", "has_air_conditioning", "thermal_inertia"]
    )
    scaler = ColumnTransformer(
        [("scaler", StandardScaler(), ["living_area_sqft"])], remainder="passthrough"
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("selector", selector),
            ("scaler", scaler),
        ]
    )

    return pipe
