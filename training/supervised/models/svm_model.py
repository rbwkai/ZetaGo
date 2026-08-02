"""Linear SVM baseline using scikit-learn.

Policy: `LinearSVC` (classification). Value: `LinearSVR` (regression, margin
target -- task 1.1b). Does not scale to the full corpus (EXECUTION_Phase1.md
task 1.7): relies on the same `--baseline-train-cap` subsampling the other
classical baselines already use, so the per-compute comparison across
classical models stays fair. The cap is a documented finding for the paper,
not something to hide.
"""

from __future__ import annotations

import numpy as np

from ..metrics import softmax_np
from .base import SupervisedModel, full_class_proba


class SVMModel(SupervisedModel):
    name = "svm"

    def __init__(self, seed: int = 42, c: float = 1.0):
        try:
            from sklearn.svm import LinearSVC, LinearSVR
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for SVMModel") from exc

        self.move_model = LinearSVC(C=c, random_state=seed, dual="auto", max_iter=5000)
        self.value_model = LinearSVR(C=c, max_iter=5000)

    @property
    def expects_flattened(self) -> bool:
        return True

    def fit(self, x_train: np.ndarray, y_move_train: np.ndarray, y_value_train: np.ndarray) -> None:
        self.move_model.fit(x_train, y_move_train)
        self.value_model.fit(x_train, y_value_train)

    def predict(self, x_val: np.ndarray):
        # LinearSVC has no predict_proba; convert its per-class decision scores
        # to a probability-like distribution via softmax for a comparable top-k API.
        scores = self.move_model.decision_function(x_val)
        if scores.ndim == 1:
            # Binary edge case: decision_function returns one score per sample.
            # Stack into (n, 2) so full_class_proba's column-scatter still works.
            scores = np.stack([-scores, scores], axis=1)
        move_proba_small = softmax_np(scores)
        move_proba = full_class_proba(move_proba_small, self.move_model.classes_, 50)
        value_pred = self.value_model.predict(x_val).astype(np.float32)
        return move_proba, value_pred
