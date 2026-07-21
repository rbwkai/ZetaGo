"""kNN baseline implemented using scikit-learn.

This wrapper uses `KNeighborsClassifier` for the policy and
`KNeighborsRegressor` for the value head. Using scikit-learn provides highly
optimized C implementations and multi-threading while keeping the simple API
used by the trainer.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .base import SupervisedModel, full_class_proba


class KNNModel(SupervisedModel):
    name = "knn"

    def __init__(self, k: int = 11, n_jobs: int = -1):
        try:
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for KNNModel") from exc

        self.k = int(k)
        self.n_jobs = int(n_jobs)
        self._clf = KNeighborsClassifier(n_neighbors=self.k, weights="uniform", n_jobs=self.n_jobs)
        self._reg = KNeighborsRegressor(n_neighbors=self.k, weights="uniform", n_jobs=self.n_jobs)

    @property
    def expects_flattened(self) -> bool:
        return True

    def fit(self, x_train: np.ndarray, y_move_train: np.ndarray, y_value_train: np.ndarray) -> None:
        X = x_train.reshape(len(x_train), -1)
        self._clf.fit(X, y_move_train)
        # value regression expects float targets
        self._reg.fit(X, y_value_train.astype(float))

    def predict(self, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xq = x_val.reshape(len(x_val), -1)
        move_proba_small = self._clf.predict_proba(Xq)
        move_proba = full_class_proba(move_proba_small, self._clf.classes_, 50)
        value_pred = self._reg.predict(Xq).astype(np.float32)
        return move_proba, value_pred
