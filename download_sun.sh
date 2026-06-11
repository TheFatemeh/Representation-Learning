#!/bin/bash
# Download ONLY the SUN397 images (the dataset skipped earlier for disk space).
# Run from MainRepo/:   bash download_sun.sh
# The SUN397 CoOp split (split_zhou_SUN397.json) was already fetched by download_datasets.sh
# and lives in TCA/splits/.
#
# Estimated disk: ~37 GB.

set -e

DATA_ROOT="$(pwd)/TCA/data"

# --- SUN397 ---
echo "=== SUN397 ==="
mkdir -p "$DATA_ROOT/sun397"
cd "$DATA_ROOT/sun397"
KAGGLE_USER=$(python3 -c "import json; d=json.load(open('/home/hpc/rlvl/rlvl168v/.kaggle/kaggle.json')); print(d['username'])")
KAGGLE_KEY=$(python3 -c "import json; d=json.load(open('/home/hpc/rlvl/rlvl168v/.kaggle/kaggle.json')); print(d['key'])")
mkdir -p /scratch/rlvl168v
wget -c --show-progress \
    --user="$KAGGLE_USER" --password="$KAGGLE_KEY" \
    "https://www.kaggle.com/api/v1/datasets/download/hiuphmnhtrung/sun397" \
    -O /scratch/rlvl168v/sun397.zip
unzip -q /scratch/rlvl168v/sun397.zip -d "$DATA_ROOT/sun397"
rm /scratch/rlvl168v/sun397.zip
# Normalize: move SUN397/ up if it landed inside a subdirectory
if [ ! -d "$DATA_ROOT/sun397/SUN397" ]; then
    FOUND=$(find "$DATA_ROOT/sun397" -type d -name "SUN397" | head -1)
    [ -n "$FOUND" ] && mv "$FOUND" "$DATA_ROOT/sun397/SUN397"
fi
echo "  Done. Expected: $DATA_ROOT/sun397/SUN397/"
