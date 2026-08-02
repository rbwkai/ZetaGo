#!/usr/bin/env bash
# Push the frozen train/val/test HDF5 splits + dataset card to Kaggle as a
# private Dataset (EXECUTION_Phase2.md task 2.0c). One-time unless the corpus
# is regenerated -- unlike sync_code.sh, this is not part of the normal
# day-to-day workflow.
#
# Usage: kaggle/sync_dataset.sh ["version message"]
set -euo pipefail

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: the 'kaggle' CLI is not installed or not on PATH (pip install kaggle)" >&2
  exit 1
fi
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "error: no Kaggle API token at ~/.kaggle/kaggle.json -- see https://www.kaggle.com/settings -> API" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if grep -q "<KAGGLE_USERNAME>" "$ROOT_DIR/kaggle/data-dataset-metadata.json"; then
  echo "error: fill in your real Kaggle username in kaggle/data-dataset-metadata.json first" >&2
  exit 1
fi

for f in train.h5 val.h5 test.h5; do
  if [ ! -f "$ROOT_DIR/data/processed/$f" ]; then
    echo "error: $ROOT_DIR/data/processed/$f is missing -- run data/build_dataset.py first" >&2
    exit 1
  fi
done

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp "$ROOT_DIR/kaggle/data-dataset-metadata.json" "$STAGING/dataset-metadata.json"
cp "$ROOT_DIR/data/processed/train.h5" "$STAGING/"
cp "$ROOT_DIR/data/processed/val.h5" "$STAGING/"
cp "$ROOT_DIR/data/processed/test.h5" "$STAGING/"
cp "$ROOT_DIR/data/processed/DATASET_CARD.md" "$STAGING/"

MESSAGE="${1:-sync $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
echo "Staged $(du -sh "$STAGING" | cut -f1) at $STAGING, pushing as: $MESSAGE"

if kaggle datasets version -p "$STAGING" -m "$MESSAGE" -t; then
  echo "Pushed a new version of zetago-dataset-7x7."
else
  echo "version failed (dataset may not exist yet) -- trying create instead..."
  kaggle datasets create -p "$STAGING" -t
fi
