from __future__ import annotations

import dataclasses

import numpy as np
import scipy
from sklearn.utils import extmath


@dataclasses.dataclass
class SVD:
    U: np.ndarray
    s: np.ndarray
    V: np.ndarray


def compute_svd(
    X: np.ndarray,
    n_components: int,
    n_iter: int,
    engine: str,
    random_state: int | None = None,
    row_weights: np.ndarray | None = None,
    column_weights: np.ndarray | None = None,
) -> SVD:
    if row_weights is not None:
        X = X * np.sqrt(row_weights[:, np.newaxis])
    if column_weights is not None:
        X = X * np.sqrt(column_weights)

    if engine == "scipy":
        U, s, V = scipy.linalg.svd(X)
        U = U[:, :n_components]
        s = s[:n_components]
        V = V[:n_components, :]
    elif engine == "sklearn":
        U, s, V = extmath.randomized_svd(
            X,
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
        )
    else:
        raise ValueError("engine has to be one of ('scipy', 'sklearn')")

    if row_weights is not None:
        U = U / np.sqrt(row_weights)[:, np.newaxis]
    if column_weights is not None:
        V = V / np.sqrt(column_weights)

    return SVD(U, s, V)
