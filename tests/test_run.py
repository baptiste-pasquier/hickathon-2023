import pickle
from pathlib import Path

from hickathon_2023.model import get_model
from hickathon_2023.utils import load_data


def test_fit_predict():
    Y_FEATURE = "energy_consumption_per_annum"
    X_train, y_train = load_data(
        X_path=Path("datasets/small/train/train_features_sent.csv"),
        y_path=Path("datasets/small/train/train_labels_sent.csv"),
        y_feature=Y_FEATURE,
    )
    X_test, _ = load_data(X_path=Path("datasets/small/test/test_features_sent.csv"))

    model = get_model()

    model.fit(X_train, y_train)

    model.predict(X_test)


def test_predict_from_trained():
    X_test, _ = load_data(X_path=Path("datasets/small/test/test_features_sent.csv"))

    trained_model_path = Path("trained_model/trained_model.pickle")
    with open(trained_model_path, "rb") as f:
        model = pickle.load(f)

    model.predict(X_test)
