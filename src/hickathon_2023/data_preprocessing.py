import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from hickathon_2023.features import FEATURES, FEATURES_DTYPES
from hickathon_2023.utils import Pipeline

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


# from hickathon_2023.features import sFEATURES_ONEHOT

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
            raise ValueError()
    return df


def get_data_preprocessor():

    selector = FeatureSelector(features=FEATURES)

    # inputer = ColumnTransformer(
    #     transformers=[
    #         (
    #             "imputer_median",
    #             SimpleImputer(strategy="median"),
    #             [
    #                 "altitude",
    #                 "building_height_ft",
    #                 "building_total_area_sqft",
    #                 "living_area_sqft",
    #                 "lowe_floor_thermal_conductivity",
    #                 "outer_wall_thermal_conductivity",
    #                 "percentage_glazed_surfaced",
    #                 "upper_floor_thermal_conductivity",
    #                 "window_heat_retention_factor",
    #                 "window_thermal_conductivity",
    #             ],
    #         ),
    #         (
    #             "imputer_0",
    #             SimpleImputer(strategy="constant", fill_value=0),
    #             [
    #                 "nb_commercial_units",
    #                 "nb_dwellings",
    #                 "nb_housing_units",
    #             ],
    #         ),
    #         (
    #             "imputer_1",
    #             SimpleImputer(strategy="constant", fill_value=1),
    #             ["building_use_type_code"],
    #         ),
    #     ],
    #     remainder="passthrough",
    # )

    # converter_dtypes = FunctionTransformer(convert_dtypes)

    # cat_pipeline = Pipeline(
    #     [
    #         (
    #             "ordinal_encoder",
    #             OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=bool),
    #         ),
    #     ]
    # )
    # categorical_preprocessing = ColumnTransformer(
    #     [("categorical_preproc", cat_pipeline, FEATURES_ONEHOT)],
    #     remainder="passthrough",
    # )

    pipe = Pipeline(
        [
            ("selector", selector),
            # ("inputer", inputer),
            # ("convert_type", converter_dtypes),
            # ("categorical", categorical_preprocessing),
        ]
    )

    return pipe
