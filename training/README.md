# Training — Quick Status

Short reference explaining the supervised training pipeline and how CLI hyperparameters are used.

Status
- Codebase contains dataset extractor (`data/`), train/val HDF5 splits (`data/processed/`),
  and a modular supervised training entrypoint at `training/train_supervised.py`.
- `run_simple.sh` is a small convenience wrapper that forwards CLI args and prefers `venv/bin/python`.

Training pipeline (short)
- Loads HDF5 splits with `load_split()` and builds features via `make_features()` for each
  `--encodings` value (allowed: 2, 4, 7).
- `create_model(name, in_channels, args, device)` instantiates one of: `logreg`, `rf`, `knn`, `cnn`.
- For each model: `fit()` is called using the training features; `predict()` is run on the val set
  and metrics saved to `results/`.

How `--epochs` and `--lr` are used
- Both `--epochs` and `--lr` are parsed in `training/supervised/trainer.py` and passed into
  `create_model(...)` via the `args` namespace.
- Only the `cnn` model uses these hyperparameters:
  - `--epochs` controls `CNNModel.epochs` and the number of training loop passes.
  - `--lr` sets the optimizer learning rate (`torch.optim.Adam(..., lr=args.lr)`).
- Baseline models (`logreg`, `rf`, `knn`) use scikit-learn estimators and do not use `--epochs`/`--lr`.
  These options are intentionally ignored for those models.

How to run
- Quick logistic-regression baseline (fast):

  ```bash
  ./run_simple.sh
  ```

- Run a CNN with 20 epochs and LR=0.0005 (example):

  ```bash
  ./run_simple.sh --model cnn --epochs 20 --lr 0.0005 --device cpu
  ```

