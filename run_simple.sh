#!/usr/bin/env bash
set -euo pipefail

# Fast simple run: logistic regression, N=2 features, small caps for speed
# Use project's venv python if available so installed deps are picked up.
PYTHON=python
if [ -x "venv/bin/python" ]; then
  PYTHON=venv/bin/python
fi

$PYTHON training/train_supervised.py \
  --model logreg \
  --encodings 2 \
  --max-train 30000 \
  --max-val 5000 \
  --baseline-train-cap 20000 \
  "$@"
