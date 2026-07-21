"""Model factory for supervised experiments.

Expose the small set of available model classes and provide a `create_model`
factory that converts a CLI-friendly name into an initialized object using
arguments from the trainer `args` namespace.
"""

from .base import SupervisedModel
from .cnn_model import CNNModel
from .knn_model import KNNModel
from .logreg_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel


def create_model(name: str, in_channels: int, args, device: str) -> SupervisedModel:
    if name == "logreg":
        return LogisticRegressionModel(seed=args.seed)
    if name == "rf":
        return RandomForestModel(n_estimators=args.rf_trees, seed=args.seed)
    if name == "knn":
        return KNNModel(k=args.knn_k)
    if name == "cnn":
        from ..losses import WeightedPolicyValueLoss

        loss_fn = WeightedPolicyValueLoss(value_weight=args.value_loss_weight)
        return CNNModel(
            in_channels=in_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
            loss_fn=loss_fn,
        )
    raise ValueError(f"unknown model: {name}")
