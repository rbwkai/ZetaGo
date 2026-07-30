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
from .svm_model import SVMModel


def create_model(name: str, in_channels: int, args, device: str, seed: int) -> SupervisedModel:
    if name == "logreg":
        return LogisticRegressionModel(seed=seed)
    if name == "rf":
        return RandomForestModel(n_estimators=args.rf_trees, seed=seed)
    if name == "knn":
        return KNNModel(k=args.knn_k)
    if name == "svm":
        return SVMModel(seed=seed, c=args.svm_c)
    if name == "cnn":
        from ..losses import WeightedPolicyValueLoss

        loss_fn = WeightedPolicyValueLoss(value_weight=args.value_loss_weight)
        return CNNModel(
            in_channels=in_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=seed,
            device=device,
            loss_fn=loss_fn,
            value_scale=args.value_scale,
        )
    raise ValueError(f"unknown model: {name}")
