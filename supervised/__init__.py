"""Supervised training package for Track A experiments.

This package contains a small, self-contained supervised training pipeline
that consumes HDF5 datasets produced by `data/build_dataset.py` and
produces model performance metrics (policy top-k, value MSE/MAE/acc, timing).

Key modules:
- `data`           : loading and simple subsampling of HDF5 splits
- `features`       : feature encodings (N=2,4,7) converting engine tensors -> model inputs
- `models`         : model implementations (logreg, knn, rf, cnn)
- `losses`         : loss wrappers for neural training
- `metrics`        : evaluation helpers and CSV/JSON export
- `trainer`        : high-level orchestration and CLI

Design notes:
- Models are small classes with `fit` and `predict` methods.
- Baseline models operate on flattened features; the CNN expects tensor-shaped inputs.
- The trainer ensures identical train/val splits and labels are used across models
	so results are directly comparable.
"""
