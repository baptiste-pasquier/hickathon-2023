import xgboost

from hickathon_2023.data_preprocessing import get_data_preprocessor
from hickathon_2023.feature_extraction import FeatureExtractor
from hickathon_2023.utils import Pipeline


def get_model() -> Pipeline:
    """Create a Scikit-Learn pipeline model.

    Returns
    -------
    Pipeline
        Pipeline model
    """
    feature_extractor = FeatureExtractor()
    data_preprocessor = get_data_preprocessor()

    regressor = xgboost.XGBRegressor(
        n_estimators=2000,
        max_depth=5,
        eta=0.2,
        subsample=0.7,
        colsample_bytree=0.8,
        max_leaves=20,
        n_jobs=-1,
    )

    model = Pipeline(
        [
            ("feature_extractor", feature_extractor),
            ("preprocessor", data_preprocessor),
            ("regressor", regressor),
        ]
    )

    return model
