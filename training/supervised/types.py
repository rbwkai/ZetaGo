"""Small shared types used by the supervised package.

`RunMetrics` captures the evaluation summary for a single experiment
(one model on one feature encoding, one seed, one Factor B data-volume level).
These are written out to CSV/JSON so you can collect and compare experiments.
"""

from dataclasses import dataclass


@dataclass
class RunMetrics:
    model: str
    encoding: int
    seed: int
    dedup: str
    data_volume_games: int
    data_volume_rows: int
    data_volume_unique: int
    move_top1: float
    move_top1_ci_lo: float
    move_top1_ci_hi: float
    move_top3: float
    move_macro_f1: float
    value_mse: float
    value_mae: float
    value_acc: float
    value_baseline_acc: float
    train_seconds: float
    infer_seconds_total: float
    infer_ms_per_sample: float
