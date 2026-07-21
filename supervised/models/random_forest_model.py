"""Random forest baseline using scikit-learn.

This wrapper trains a `RandomForestClassifier` for the policy and a
`RandomForestRegressor` for the value. Using sklearn's RF gives a robust and
well-optimized baseline that behaves like a standard random forest.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .base import SupervisedModel


class RandomForestModel(SupervisedModel):
    name = "rf"

    def __init__(self, n_estimators: int = 100, max_features: str = "sqrt", seed: int = 42, n_jobs: int = -1):
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for RandomForestModel") from exc

        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)

        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.reg = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )

    @property
    def expects_flattened(self) -> bool:
        return True

    def fit(self, x_train: np.ndarray, y_move_train: np.ndarray, y_value_train: np.ndarray) -> None:
        X = x_train.reshape(len(x_train), -1)
        self.clf.fit(X, y_move_train)
        self.reg.fit(X, y_value_train.astype(float))

    def predict(self, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xq = x_val.reshape(len(x_val), -1)
        move_proba = self.clf.predict_proba(Xq)
        # sklearn returns full proba with columns matching `classes_`; ensure
        # caller expects 50-wide actions (trainer uses argmax/topk only)
        # If some classes are missing, pad to width 50.
        if move_proba.shape[1] < 50:
            from .base import full_class_proba

            move_proba = full_class_proba(move_proba, self.clf.classes_, 50)

        value_pred = self.reg.predict(Xq).astype(np.float32)
        return move_proba, value_pred
