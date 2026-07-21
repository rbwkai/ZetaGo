"""Small shared types used by the supervised package.

`RunMetrics` captures the evaluation summary for a single experiment
(one model on one feature encoding). These are written out to CSV/JSON
so you can collect and compare experiments.
"""

from dataclasses import dataclass


@dataclass
class RunMetrics:
    model: str
    encoding: int
    move_top1: float
    move_top3: float
    value_mse: float
    value_mae: float
    value_acc: float
    train_seconds: float
    infer_seconds_total: float
    infer_ms_per_sample: float
