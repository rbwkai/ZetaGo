"""Logistic regression baseline using scikit-learn.

This thin wrapper trains two multinomial logistic regression models: one for
policy (predicting a move index) and one for value. Scikit-learn is an
optional dependency for the project; if it's not installed this model will
raise at construction time.
"""

from __future__ import annotations

import numpy as np

from .base import SupervisedModel, full_class_proba


class LogisticRegressionModel(SupervisedModel):
    name = "logreg"

    def __init__(self, seed: int = 42):
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for logistic regression") from exc

        self.move_model = LogisticRegression(
            max_iter=500,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=seed,
        )
        # value modeled as a multinomial regressor here for simplicity
        self.value_model = LogisticRegression(
            max_iter=500,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=seed,
        )

    def fit(self, x_train: np.ndarray, y_move_train: np.ndarray, y_value_train: np.ndarray) -> None:
        self.move_model.fit(x_train, y_move_train)
        self.value_model.fit(x_train, y_value_train)

    def predict(self, x_val: np.ndarray):
        move_proba_small = self.move_model.predict_proba(x_val)
        move_proba = full_class_proba(move_proba_small, self.move_model.classes_, 50)
        value_pred = self.value_model.predict(x_val).astype(np.float32)
        return move_proba, value_pred
