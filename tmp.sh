#!/bin/bash
# Remaining downloads: SUN397 via Kaggle (wget streaming to avoid MemoryError)

DATA_ROOT="$(pwd)/TCA/data"

# --- SUN397 ---
echo "[9/10] SUN397"
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

echo ""
echo "=== Done ==="
