#!/usr/bin/env bash
set -euo pipefail

# Fast simple run: logistic regression, N=2 features, small caps for speed
python train_supervised.py \
  --model logreg \
  --encodings 2 \
  --max-train 30000 \
  --max-val 5000 \
  --baseline-train-cap 20000
