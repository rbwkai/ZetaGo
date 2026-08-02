#!/usr/bin/env bash
# Push the training/eval code -- not the dataset, not the katago binaries,
# not the GUI -- to Kaggle as a private Dataset (EXECUTION_Phase2.md task
# 2.0b). Run this after ANY local change to a file it stages, before the next
# Kaggle session; there is no automatic sync (EXECUTION_Phase2.md SS2.5).
#
# Usage: kaggle/sync_code.sh ["version message"]
# First run creates the dataset; every run after that pushes a new version.
# Fill in a real Kaggle username in kaggle/code-dataset-metadata.json first.
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
if grep -q "<KAGGLE_USERNAME>" "$ROOT_DIR/kaggle/code-dataset-metadata.json"; then
  echo "error: fill in your real Kaggle username in kaggle/code-dataset-metadata.json first" >&2
  exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp "$ROOT_DIR/kaggle/code-dataset-metadata.json" "$STAGING/dataset-metadata.json"

# Allow-list: everything training/eval actually imports, nothing else.
# environment/katago/ (131 MB of binary + network) and environment/gui|assets/ (pygame,
# unused on the training path) are deliberately excluded -- see
# EXECUTION_Phase2.md SS2.2 for why.
ITEMS=(
  environment/engine
  training
  eval
  "data/dataset generation/build_dataset.py"
  "data/dataset generation/check_dataset.py"
  "data/dataset generation/export_csv.py"
  data/sgf_reader.py
  data/analysis/plot_class_distribution.py
  data/__init__.py
  tests
  conftest.py
  pytest.ini
  requirements.txt
)

for item in "${ITEMS[@]}"; do
  src="$ROOT_DIR/$item"
  dest="$STAGING/$item"
  mkdir -p "$(dirname "$dest")"
  if [ -d "$src" ]; then
    cp -r "$src" "$dest"
  elif [ -f "$src" ]; then
    cp "$src" "$dest"
  else
    echo "warning: $item not found locally, skipping" >&2
  fi
done

find "$STAGING" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

MESSAGE="${1:-sync $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
echo "Staged $(du -sh "$STAGING" | cut -f1) at $STAGING, pushing as: $MESSAGE"

# -r zip: without it the CLI silently *skips* every subdirectory (its default
# dir-mode is "skip", confirmed the hard way -- environment/, training/,
# eval/, data/, tests/ all got dropped on a first attempt with no error).
# -t: don't let the CLI "helpfully" convert any file to CSV.
if kaggle datasets version -p "$STAGING" -m "$MESSAGE" -r zip -t; then
  echo "Pushed a new version of zetago-code."
else
  echo "version failed (dataset may not exist yet) -- trying create instead..."
  kaggle datasets create -p "$STAGING" -r zip -t
fi
