import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from hickathon_2023.features import FEATURES, FEATURES_ONEHOT
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


# class DataProcessing(BaseEstimator):
#     def fit(self, X, y):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         return X


def get_data_preprocessor():

    # preprocess = DataProcessing()
    selector = FeatureSelector(features=FEATURES)
    # scaler = ColumnTransformer(
    #     [("scaler", StandardScaler(), ["living_area_sqft"])], remainder="passthrough"
    # )

    cat_pipeline = Pipeline(
        [
            ("ordinal_encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    categorical_preprocessing = ColumnTransformer(
        [("categorical_preproc", cat_pipeline, FEATURES_ONEHOT)],
        remainder="passthrough",
    )

    pipe = Pipeline(
        [
            # ("preprocess", preprocess),
            ("selector", selector),
            ("categorical", categorical_preprocessing)
            # ("scaler", scaler),
        ]
    )

    return pipe
