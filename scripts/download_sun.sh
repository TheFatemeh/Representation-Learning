#!/bin/bash
# Download ONLY the SUN397 images (the dataset skipped earlier for disk space).
# Run from MainRepo/:   bash download_sun.sh
# The SUN397 CoOp split (split_zhou_SUN397.json) was already fetched by download_datasets.sh
# and lives in TCA/splits/.
#
# Estimated disk: ~37 GB.

set -e

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"

DATA_ROOT=$MAIN/TCA/data

# --- SUN397 ---
echo "=== SUN397 ==="
mkdir -p "$DATA_ROOT/sun397"
cd "$DATA_ROOT/sun397"
KAGGLE_USER=$(python3 -c "import json; d=json.load(open('$HOME/.kaggle/kaggle.json')); print(d['username'])")
KAGGLE_KEY=$(python3 -c "import json; d=json.load(open('$HOME/.kaggle/kaggle.json')); print(d['key'])")
mkdir -p /scratch/$USER
wget -c --show-progress \
    --user="$KAGGLE_USER" --password="$KAGGLE_KEY" \
    "https://www.kaggle.com/api/v1/datasets/download/hiuphmnhtrung/sun397" \
    -O /scratch/$USER/sun397.zip
unzip -q /scratch/$USER/sun397.zip -d "$DATA_ROOT/sun397"
rm /scratch/$USER/sun397.zip
# Normalize so images end at data/sun397/SUN397/<letter>/<category>/...
# (the Kaggle zip double-nests as sun397/SUN397/SUN397/...)
if [ -d "$DATA_ROOT/sun397/SUN397/SUN397" ]; then
    mv "$DATA_ROOT/sun397/SUN397" "$DATA_ROOT/sun397/SUN397_outer"
    mv "$DATA_ROOT/sun397/SUN397_outer/SUN397" "$DATA_ROOT/sun397/SUN397"
    rm -rf "$DATA_ROOT/sun397/SUN397_outer"
elif [ ! -d "$DATA_ROOT/sun397/SUN397" ]; then
    FOUND=$(find "$DATA_ROOT/sun397" -type d -name "SUN397" | head -1)
    [ -n "$FOUND" ] && mv "$FOUND" "$DATA_ROOT/sun397/SUN397"
fi
echo "  Done. Expected: $DATA_ROOT/sun397/SUN397/"
