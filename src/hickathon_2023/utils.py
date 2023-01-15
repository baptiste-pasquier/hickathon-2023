import logging

import numpy as np
import pandas as pd
import sklearn.impute
import sklearn.pipeline
from sklearn.base import clone
from sklearn.pipeline import _fit_transform_one
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory

log = logging.getLogger(__name__)


def load_data(X_path, y_path=None, y_feature=None):
    df_X = pd.read_csv(X_path, low_memory=False)

    for column in df_X.columns:
        if df_X[column].dtype == np.int64:
            df_X[column] = df_X[column].astype(np.int16)
        if df_X[column].dtype == np.float64:
            df_X[column] = df_X[column].astype(np.float32)

    if y_path:
        if y_feature is None:
            raise ValueError("y_pred_name must be defined")
        y = pd.read_csv(y_path)[y_feature]
    else:
        y = None

    return df_X, y


class Pipeline(sklearn.pipeline.Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)

            # Logging
            log.info(f"Dataset before {name} : shape {X.shape}")
            if isinstance(X, pd.DataFrame):
                log.debug(f"Dataset before {name} : features {X.columns.tolist()}")
            log.debug(f"Applyging {name}")

            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)

            # Logging
            log.info(f"Dataset after {name} : shape {X.shape}")
            if isinstance(X, pd.DataFrame):
                log.debug(f"Dataset after {name} : features {X.columns.tolist()}")
        return X


# class SimpleImputer(sklearn.impute.SimpleImputer):
#     """docstring for SimpleImputer."""

#     def __init__(
#         self,
#         *,
#         missing_values=np.nan,
#         strategy="mean",
#         fill_value=None,
#         verbose="deprecated",
#         copy=True,
#         add_indicator=False,
#     ):
#         super(SimpleImputer, self).__init__(
#             missing_values=missing_values,
#             strategy=strategy,
#             fill_value=fill_value,
#             verbose=verbose,
#             copy=copy,
#             add_indicator=add_indicator,
#         )

#         def transform(self, X):
#             X_inputed = super().transform(X)
#             features_names = super().get_feature_names_out()
#             return pd.DataFrame(X_inputed, columns=features_names)
