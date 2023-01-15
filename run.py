import argparse
import logging
import pickle
import warnings
from pathlib import Path

import pandas as pd

from hickathon_2023.model import get_model
from hickathon_2023.utils import load_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--predict", action="store_true", help="Predict on the test dataset"
    )
    parser.add_argument("--small", action="store_true", help="Use the small dataset")
    args = parser.parse_args()

    if not (args.train or args.predict):
        parser.error("Error: No actions requested")

    logging.basicConfig()
    logging.getLogger("hickathon_2023").setLevel(logging.INFO)
    log = logging.getLogger("run")
    log.setLevel(logging.INFO)

    Y_FEATURE = "energy_consumption_per_annum"

    # Loading datasets
    if args.small:
        log.info("Loading big datasets")
        datasets_path = Path("datasets/small/")
        trained_model_path = Path("trained_model/small/trained_model.pickle")
    else:
        log.info("Loading small datasets")
        datasets_path = "datasets/"
        trained_model_path = Path("/trained_model/trained_model.pickle")

    # Creating trained_model folder
    trained_model_path.parents[0].mkdir(parents=True, exist_ok=True)

    # Training model
    if args.train:
        X_path = Path(datasets_path, "train/train_features_sent.csv")
        y_path = Path(datasets_path, "train/train_labels_sent.csv")
        log.info(f"Loading {X_path}")
        log.info(f"Loading {y_path}")
        X_train, y_train = load_data(
            X_path=X_path,
            y_path=y_path,
            y_feature=Y_FEATURE,
        )

        model = get_model()

        log.info("Fitting model")
        model.fit(X_train, y_train)

        log.info(f"Saving model to {trained_model_path}")
        with open(trained_model_path, "wb") as f:
            pickle.dump(model, f)

    # Predicting
    if args.predict:
        X_path = Path(datasets_path, "test/test_features_sent.csv")
        log.info(f"Loading {X_path}")
        X_test, _ = load_data(X_path=X_path)

        if not trained_model_path.exists():
            raise Exception("No trained model found. Please run --train command")
        with open(trained_model_path, "rb") as f:
            model = pickle.load(f)

        log.info("Calculating predictions")
        y_test_pred = model.predict(X_test)

        submission_template_path = Path(datasets_path, "sample_submission_sent.csv")
        log.info(f"Loading predictions template from {submission_template_path}")
        submission_df = pd.read_csv(submission_template_path)

        # Checking index
        assert (submission_df["level_0"] != X_test["level_0"]).sum() == 0

        submission_df[Y_FEATURE] = y_test_pred
        submission_path = Path(datasets_path, "submission_sent.csv")
        log.info(f"Exporting predictions to {submission_path}")
        submission_df.to_csv(submission_path, index=False)
