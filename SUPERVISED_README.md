Quick guide — supervised Track A (simple)

Goal: Train a simple baseline quickly and compare later.

1) Install dependencies (recommended inside `venv`):

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

2) Fast, minimal run (logistic regression, N=2 features):

```bash
# small-ish baseline run (fast)
python train_supervised.py --model logreg --encodings 2 --max-train 30000 --max-val 5000 --baseline-train-cap 20000
```

3) Full comparison (all models, all encodings):

```bash
python train_supervised.py --model all --encodings 2 4 7
```

4) Quick CNN test (if you have GPU):

```bash
python train_supervised.py --model cnn --encodings 7 --epochs 4 --device cuda
```

Files produced:
- `results/supervised_track_a_metrics.csv`
- `results/supervised_track_a_metrics.json`

Notes:
- If `h5py` or `scikit-learn`/`torch` are missing, install via `pip install -r requirements.txt`.
- Use `--max-train` and `--max-val` to cap rows for quick debugging.
