import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from hickathon_2023.features import FEATURES, FEATURES_DTYPES, FEATURES_ONEHOT
from hickathon_2023.utils import Pipeline

# from hickathon_2023.utils import SimpleImputer


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


def convert_dtypes(df):
    for column, dtype in FEATURES_DTYPES.items():
        if dtype == "np.int8":
            df[column] = df[column].astype(np.int8)
        elif dtype == "np.int16":
            df[column] = df[column].astype(np.int16)
        else:
            raise ValueError(column)
    return df


def rename_columns(df):
    columns = df.columns.to_list()
    for i in range(len(columns)):
        for elem in [
            "imputer_median__",
            "imputer_0__",
            "imputer_1__",
            "remainder__",
            "onehot__",
        ]:
            columns[i] = columns[i].replace(elem, "")
    df.columns = columns

    return df


def get_data_preprocessor():

    selector = FeatureSelector(features=FEATURES)

    inputer = ColumnTransformer(
        transformers=[
            (
                "imputer_median",
                SimpleImputer(strategy="median").set_output(transform="pandas"),
                [
                    "altitude",
                    "building_height_ft",
                    "building_total_area_sqft",
                    "living_area_sqft",
                    "living_to_building_area_ratio",
                    "lowe_floor_thermal_conductivity",
                    "outer_wall_thermal_conductivity",
                    "percentage_glazed_surfaced",
                    "upper_floor_thermal_conductivity",
                    "wall_area_by_conductivity",
                    "window_heat_retention_factor",
                    "window_thermal_conductivity",
                ],
            ),
            (
                "imputer_0",
                SimpleImputer(strategy="constant", fill_value=0).set_output(
                    transform="pandas"
                ),
                [
                    "nb_commercial_units",
                    "nb_dwellings",
                    "nb_housing_units",
                ],
            ),
            (
                "imputer_1",
                SimpleImputer(strategy="constant", fill_value=1).set_output(
                    transform="pandas"
                ),
                ["building_use_type_code", "nb_meters", "nb_units_total"],
            ),
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")

    converter_dtypes = FunctionTransformer(convert_dtypes).set_output(
        transform="pandas"
    )

    categorical_preprocessing = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=bool
                ).set_output(transform="pandas"),
                FEATURES_ONEHOT,
            )
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")

    column_renamer = FunctionTransformer(rename_columns).set_output(transform="pandas")

    pipe = Pipeline(
        [
            ("selector", selector),
            ("inputer", inputer),
            ("renamer_1", column_renamer),
            ("convert_type", converter_dtypes),
            ("onehot", categorical_preprocessing),
            ("renamer_2", column_renamer),
        ]
    )

    return pipe
