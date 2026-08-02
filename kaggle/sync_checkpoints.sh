#!/usr/bin/env bash
# Push a downloaded Kaggle notebook-version's results/+checkpoints/ output as
# a private zetago-checkpoints Dataset. This is THE reliable interruption-
# recovery path (EXECUTION_Phase2.md, corrected 31 July 2026): "revert to a
# saved notebook version" does NOT reliably restore /kaggle/working -- a real
# attached Dataset does, every time, via kaggle_trainer.ipynb's Step 2.
#
# Usage: kaggle/sync_checkpoints.sh <folder containing results/ and/or checkpoints/> ["message"]
# The folder is wherever you downloaded/extracted a notebook version's Output
# to on this machine -- it must directly contain a results/ and/or
# checkpoints/ subdirectory (not nested further).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: kaggle/sync_checkpoints.sh <folder containing results/ and/or checkpoints/> [\"message\"]" >&2
  exit 1
fi
SRC_DIR="$1"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: the 'kaggle' CLI is not installed or not on PATH (pip install kaggle)" >&2
  exit 1
fi
if [ ! -f "$HOME/.kaggle/access_token" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "error: no Kaggle credentials found at ~/.kaggle/access_token or kaggle.json" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if grep -q "<KAGGLE_USERNAME>" "$ROOT_DIR/kaggle/checkpoints-dataset-metadata.json"; then
  echo "error: fill in your real Kaggle username in kaggle/checkpoints-dataset-metadata.json first" >&2
  exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp "$ROOT_DIR/kaggle/checkpoints-dataset-metadata.json" "$STAGING/dataset-metadata.json"

found_any=0
for name in results checkpoints; do
  if [ -d "$SRC_DIR/$name" ]; then
    cp -r "$SRC_DIR/$name" "$STAGING/$name"
    found_any=1
  fi
done
if [ "$found_any" -eq 0 ]; then
  echo "error: neither '$SRC_DIR/results' nor '$SRC_DIR/checkpoints' exists under $SRC_DIR -- did you point at the right download?" >&2
  exit 1
fi

MESSAGE="${2:-checkpoints sync $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
echo "Staged $(du -sh "$STAGING" | cut -f1) at $STAGING, pushing as: $MESSAGE"

if kaggle datasets version -p "$STAGING" -m "$MESSAGE" -r zip -t; then
  echo "Pushed a new version of zetago-checkpoints."
else
  echo "version failed (dataset may not exist yet) -- trying create instead..."
  kaggle datasets create -p "$STAGING" -r zip -t
fi
