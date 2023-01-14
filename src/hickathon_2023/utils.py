import pandas as pd


def load_data(X_path, y_path=None, y_feature=None):
    df_X = pd.read_csv(X_path, low_memory=False)

    if y_path:
        if y_feature is None:
            raise ValueError("y_pred_name must be defined")
        y = pd.read_csv(y_path)[y_feature]
    else:
        y = None

    return df_X, y
