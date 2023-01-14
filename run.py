import argparse
import logging

import pandas as pd

from hickathon_2023.model import get_model
from hickathon_2023.utils import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true")  # on/off flag
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger("hickathon_2023").setLevel(logging.INFO)
    log = logging.getLogger("run")
    log.setLevel(logging.INFO)

    Y_FEATURE = "energy_consumption_per_annum"

    log.info("Loading small datasets")
    if args.small:
        path = "datasets/small/"
    else:
        path = "datasets/"
    X_train, y_train = load_data(
        X_path=path + "train/train_features_sent.csv",
        y_path=path + "train/train_labels_sent.csv",
        y_feature=Y_FEATURE,
    )
    X_test, _ = load_data(X_path=path + "test/test_features_sent.csv")

    model = get_model()

    log.info("Fitting model")
    model.fit(X_train, y_train)

    log.info("Predicting")
    y_test_pred = model.predict(X_test)

    log.info("Exporting predictions")
    submission_df = pd.read_csv(path + "sample_submission_sent.csv")
    submission_df[Y_FEATURE] = y_test_pred
    submission_df.to_csv(path + "submission_sent.csv", index=True)
