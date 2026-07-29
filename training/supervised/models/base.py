"""Abstract base for supervised models.

Concrete model implementations should implement `fit` and `predict` and
advertise whether they expect flattened 1D feature vectors (`expects_flattened`).

Helper `full_class_proba` expands class-indexed probability vectors returned
by some baselines into a dense NxC matrix where C is the full action space
(e.g. 49 board + pass).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SupervisedModel(ABC):
    name = "base"

    @property
    def expects_flattened(self) -> bool:
        # Baseline models typically work with 1D feature vectors.
        return True

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_move_train: np.ndarray, y_value_train: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (move_proba, value_pred)."""
        raise NotImplementedError


def full_class_proba(proba: np.ndarray, classes: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((proba.shape[0], num_classes), dtype=np.float32)
    out[:, classes.astype(np.int64)] = proba.astype(np.float32)
    return out
