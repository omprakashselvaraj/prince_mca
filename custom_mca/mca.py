"""Multiple Correspondence Analysis (MCA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.utils

from . import utils
from .ca import CA


class MCA(CA, sklearn.base.TransformerMixin):
    def __init__(
        self,
        n_components=2,
        n_iter=10,
        copy=True,
        check_input=True,
        random_state=None,
        engine="sklearn",
        one_hot=True,
        one_hot_prefix_sep="__",
        one_hot_columns_to_drop=None,
        correction=None,
    ):
        if correction is not None:
            if correction not in {"benzecri", "greenacre"}:
                raise ValueError("correction must be either 'benzecri' or 'greenacre' if provided.")
            if not one_hot:
                raise ValueError(
                    "correction can only be applied when one_hot is True because the number "
                    "of original variables is needed to apply the correction."
                )

        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine,
        )
        self.one_hot = one_hot
        self.one_hot_prefix_sep = one_hot_prefix_sep
        self.one_hot_columns_to_drop = one_hot_columns_to_drop
        self.correction = correction

    def _prepare(self, X):
        if self.one_hot:
            X = pd.get_dummies(X, columns=X.columns, prefix_sep=self.one_hot_prefix_sep)
            if self.one_hot_columns_to_drop is not None:
                X = X.drop(columns=self.one_hot_columns_to_drop, errors="ignore")
            if (one_hot_columns_ := getattr(self, "one_hot_columns_", None)) is not None:
                X = X.reindex(columns=one_hot_columns_.union(X.columns), fill_value=False)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.arange(self.n_components_)

    @property
    def eigenvalues_(self):
        eigenvalues = super().eigenvalues_
        if self.correction in {"benzecri", "greenacre"}:
            K = self.K_
            return np.array(
                [(K / (K - 1) * (eig - 1 / K)) ** 2 if eig > 1 / K else 0 for eig in eigenvalues]
            )
        return eigenvalues

    @property
    @utils.check_is_fitted
    def percentage_of_variance_(self):
        if self.correction == "benzecri":
            eigenvalues = self.eigenvalues_
            return 100 * eigenvalues / eigenvalues.sum()
        if self.correction == "greenacre":
            eigenvalues = super().eigenvalues_
            benzecris = self.eigenvalues_
            K, J = (self.K_, self.J_)
            average_inertia = (K / (K - 1)) * ((eigenvalues**2).sum() - (J - K) / K**2)
            return 100 * benzecris / average_inertia
        return super().percentage_of_variance_

    @utils.check_is_dataframe_input
    def fit(self, X, y=None):
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])

        self.K_ = X.shape[1]
        one_hot = self._prepare(X)
        self.one_hot_columns_ = one_hot.columns
        self.J_ = one_hot.shape[1]
        super().fit(one_hot)
        self.n_components_ = len(self.eigenvalues_)
        return self

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_coordinates(self, X):
        return super().row_coordinates(self._prepare(X))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def row_cosine_similarities(self, X):
        oh = self._prepare(X)
        return super()._row_cosine_similarities(X=oh, F=super().row_coordinates(oh))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_coordinates(self, X):
        return super().column_coordinates(self._prepare(X))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def column_cosine_similarities(self, X):
        oh = self._prepare(X)
        return super()._column_cosine_similarities(X=oh, G=super().column_coordinates(oh))

    @utils.check_is_dataframe_input
    @utils.check_is_fitted
    def transform(self, X):
        if self.check_input:
            sklearn.utils.check_array(X, dtype=[str, "numeric"])
        return self.row_coordinates(X)
