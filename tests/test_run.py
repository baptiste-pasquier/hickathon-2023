import pandas as pd

from hickathon_2023.model import get_model
from hickathon_2023.utils import load_data


def test_run_small():
    Y_FEATURE = "energy_consumption_per_annum"
    X_train, y_train = load_data(
        X_path="datasets/small/train/train_features_sent.csv",
        y_path="datasets/small/train/train_labels_sent.csv",
        y_feature=Y_FEATURE,
    )
    X_test, _ = load_data(X_path="datasets/small/test/test_features_sent.csv")

    model = get_model()

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    submission_df = pd.read_csv("datasets/small/sample_submission_sent.csv")
    submission_df[Y_FEATURE] = y_test_pred
    submission_df.to_csv("datasets/small/submission_sent.csv", index=True)
