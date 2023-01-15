from hickathon_2023.model import get_model
from hickathon_2023.utils import load_data


def test_fit_predict_small():
    Y_FEATURE = "energy_consumption_per_annum"
    X_train, y_train = load_data(
        X_path="datasets/small/train/train_features_sent.csv",
        y_path="datasets/small/train/train_labels_sent.csv",
        y_feature=Y_FEATURE,
    )
    X_test, _ = load_data(X_path="datasets/small/test/test_features_sent.csv")

    model = get_model()

    model.fit(X_train, y_train)

    model.predict(X_test)
