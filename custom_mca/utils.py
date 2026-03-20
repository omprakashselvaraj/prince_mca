from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from sklearn.utils import validation


def check_is_fitted(method):
    @functools.wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        validation.check_is_fitted(self)
        return method(self, *method_args, **method_kwargs)

    return _impl


def check_is_dataframe_input(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        X = args[1]
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"The X argument must be a pandas DataFrame, but got {type(X).__name__}"
            )
        return func(*args, **kwargs)

    return wrapper


def make_labels_and_names(X):
    if isinstance(X, pd.DataFrame):
        row_label = X.index.name if X.index.name else "Rows"
        row_names = X.index.tolist()
        col_label = X.columns.name if X.columns.name else "Columns"
        col_names = X.columns.tolist()
    else:
        row_label = "Rows"
        row_names = list(range(X.shape[0]))
        col_label = "Columns"
        col_names = list(range(X.shape[1]))

    return row_label, row_names, col_label, col_names


class EigenvaluesMixin:
    @property
    @check_is_fitted
    def percentage_of_variance_(self):
        return 100 * self.eigenvalues_ / self.total_inertia_

    @property
    @check_is_fitted
    def cumulative_percentage_of_variance_(self):
        return np.cumsum(self.percentage_of_variance_)

    @property
    @check_is_fitted
    def _eigenvalues_summary(self):
        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues_,
                "% of variance": self.percentage_of_variance_,
                "% of variance (cumulative)": self.cumulative_percentage_of_variance_,
            },
            index=pd.RangeIndex(0, len(self.eigenvalues_), name="component"),
        )

    @property
    def eigenvalues_summary(self):
        summary = self._eigenvalues_summary.copy()
        summary["% of variance"] /= 100
        summary["% of variance (cumulative)"] /= 100
        summary["eigenvalue"] = summary["eigenvalue"].map("{:,.3f}".format)
        summary["% of variance"] = summary["% of variance"].map("{:.2%}".format)
        summary["% of variance (cumulative)"] = summary["% of variance (cumulative)"].map(
            "{:.2%}".format
        )
        summary.index.name = "component"
        return summary
