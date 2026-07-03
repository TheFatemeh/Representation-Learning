#!/bin/bash
# Download ONLY the EuroSAT dataset and make it ready to use in this (untouched) TCA repo.
#
# Run from the TCA-StressTest/ directory:
#   bash download_eurosat.sh
#
# It performs three things and NO source-code edits:
#   1. downloads + unzips the EuroSAT images          -> data/eurosat/2750/<Class>/*.jpg
#   2. symlinks the class dirs to where the loader looks -> data/eurosat/<Class>  -> 2750/<Class>
#   3. places the CoOp split file at the repo root        -> ./split_zhou_EuroSat.json
#
# The stock loader (datasets/eurosat.py) reads split_path="split_zhou_EuroSat.json"
# relative to the run directory and prefixes image_dir=data/eurosat to each entry
# (e.g. "AnnualCrop/AnnualCrop_1048.jpg"). The two filesystem fixes above satisfy both,
# so `python runner.py --datasets eurosat` works with no code change.
#
# Disk: ~2 GB.

set -e

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
DATA_ROOT="$REPO/data/eurosat"
SPLIT_DST="$REPO/split_zhou_EuroSat.json"

echo "=== [1/3] EuroSAT images ==="
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
if [ -d "2750" ]; then
    echo "  2750/ already present, skipping download."
else
    wget -c --show-progress "https://madm.dfki.de/files/sentinel/EuroSAT.zip" -O EuroSAT.zip 2>/dev/null || \
    wget -c --show-progress "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip" -O EuroSAT.zip
    unzip -q EuroSAT.zip
    rm -f EuroSAT.zip
fi
echo "  Done -> $DATA_ROOT/2750/<Class>/"

echo "=== [2/3] class-dir symlinks (loader expects data/eurosat/<Class>/) ==="
cd "$DATA_ROOT"
for d in 2750/*/; do
    ln -sfn "2750/$(basename "$d")" "$(basename "$d")"
done
echo "  Done -> data/eurosat/<Class> -> 2750/<Class>"

echo "=== [3/3] CoOp split file (./split_zhou_EuroSat.json) ==="
# Self-contained: fetch the split straight into this repo, never reaching into a sibling.
# gdown ships with the TTA conda env; pass --env-gdown to point at another binary if needed.
if [ -f "$SPLIT_DST" ]; then
    echo "  Already present, skipping."
else
    if command -v gdown >/dev/null 2>&1; then GDOWN=gdown; \
    else GDOWN="$(ls "$HOME"/.conda/envs/TTA/bin/gdown 2>/dev/null | head -1)"; fi
    [ -n "$GDOWN" ] || { echo "  ERROR: gdown not found (activate the TTA env)"; exit 1; }
    "$GDOWN" 1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o -O "$SPLIT_DST"
    echo "  Downloaded via gdown"
fi

echo ""
echo "=== EuroSAT ready ==="
echo "  images: $DATA_ROOT/<Class>/  (8100 test images across 10 classes)"
echo "  split : $SPLIT_DST"
echo "  test  : python runner.py --datasets eurosat --data-root data/"
