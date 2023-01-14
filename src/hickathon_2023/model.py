from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from hickathon_2023.data_preprocessing import get_data_preprocessor
from hickathon_2023.feature_extraction import FeatureExtractor


def get_model():

    feature_extractor = FeatureExtractor()
    data_preprocessor = get_data_preprocessor()
    regressor = LinearRegression()

    model = Pipeline(
        [
            ("feature_extractor", feature_extractor),
            ("preprocessor", data_preprocessor),
            ("regressor", regressor),
        ]
    )

    return model
