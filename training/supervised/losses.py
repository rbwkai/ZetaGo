"""Loss wrappers used by neural models.

`LossBase` defines a small callable interface so different loss compositions
can be swapped into the `CNNModel` when training. The provided
`WeightedPolicyValueLoss` combines a cross-entropy policy loss with MSE
value loss and applies a scalar weight to the value term.

This keeps the training loop minimal while allowing configurable loss
weights for ablations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LossBase(ABC):
    @abstractmethod
    def __call__(self, policy_logits, value_pred, y_move, y_value):
        raise NotImplementedError


class WeightedPolicyValueLoss(LossBase):
    def __init__(self, value_weight: float = 0.5):
        self.value_weight = value_weight

    def __call__(self, policy_logits, value_pred, y_move, y_value):
        try:
            import torch.nn.functional as F
        except Exception as exc:
            raise RuntimeError("PyTorch is required for WeightedPolicyValueLoss") from exc

        loss_policy = F.cross_entropy(policy_logits, y_move)
        loss_value = F.mse_loss(value_pred, y_value)
        return loss_policy + self.value_weight * loss_value
