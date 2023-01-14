import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def get_data_preprocessor():

    selector = FeatureSelector(features=["year", "has_air_conditioning"])
    scaler = ColumnTransformer(
        [("scaler", StandardScaler(), ["year"])], remainder="passthrough"
    )

    pipe = Pipeline(
        [
            ("selector", selector),
            ("scaler", scaler),
        ]
    )

    return pipe
