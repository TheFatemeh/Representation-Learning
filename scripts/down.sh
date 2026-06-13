#!/bin/bash
# Download all 10 CD-benchmark datasets + CoOp split files for TCA reproduction.
#
# Run from MainRepo/:
#   bash download_datasets.sh
#
# After this script, your layout should be:
#   MainRepo/TCA/split_zhou_*.json        <- CoOp split files
#   MainRepo/TCA/data/caltech101/...
#   MainRepo/TCA/data/dtd/...
#   ... etc.
#
# Estimated disk: ~50 GB total

set -e

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"

DATA_ROOT=$MAIN/TCA/data
SPLIT_DIR=$MAIN/TCA
GDOWN=gdown
KAGGLE=kaggle

echo ""
echo "=== Step 2: Image downloads ==="



# --- Stanford Cars ---
echo "[8/10] Stanford Cars"
# Clean up old parquet download
rm -rf "$DATA_ROOT/stanford_cars"
mkdir -p "$DATA_ROOT/stanford_cars"
cd "$DATA_ROOT/stanford_cars"
$KAGGLE datasets download -d rickyyyyyyy/torchvision-stanford-cars -p "$DATA_ROOT/stanford_cars"
unzip -q torchvision-stanford-cars.zip -d "$DATA_ROOT/stanford_cars"
rm torchvision-stanford-cars.zip
# Find where cars_test landed and move it directly under DATA_ROOT if nested
if [ ! -d "$DATA_ROOT/cars_test" ]; then
    find "$DATA_ROOT/stanford_cars" -type d -name "cars_test" | head -1 | xargs -I{} cp -r {} "$DATA_ROOT/"
fi
echo "  Done. Expected: $DATA_ROOT/cars_test/ and $DATA_ROOT/cars_train/"


# --- UCF101 ---
echo "[10/10] UCF101 (mid-frames)"
mkdir -p "$DATA_ROOT/ucf101"
cd "$DATA_ROOT/ucf101"
$GDOWN 10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O -O UCF-101-midframes.zip
unzip -q UCF-101-midframes.zip
echo "  Done. Expected: $DATA_ROOT/ucf101/UCF-101-midframes/"

echo ""
echo "=== All downloads complete ==="
echo "Images:      $DATA_ROOT"
echo "Split files: $SPLIT_DIR"
