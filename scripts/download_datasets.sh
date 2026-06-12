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
mkdir -p "$DATA_ROOT"

echo "=== Step 1: CoOp split files ==="
cd "$SPLIT_DIR"
$GDOWN 1hyarUivQE36mY6jSomru6Fjd-JzwcCzN -O split_zhou_Caltech101.json
$GDOWN 1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs  -O split_zhou_OxfordPets.json
$GDOWN 1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT  -O split_zhou_StanfordCars.json
$GDOWN 1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT  -O split_zhou_OxfordFlowers.json
$GDOWN 1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0  -O cat_to_name.json
$GDOWN 1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl  -O split_zhou_Food101.json
$GDOWN 1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq  -O split_zhou_SUN397.json
$GDOWN 1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x  -O split_zhou_DescribableTextures.json
$GDOWN 1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o  -O split_zhou_EuroSat.json
$GDOWN 1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y  -O split_zhou_UCF101.json
echo "  Done. Split files saved to: $SPLIT_DIR"

echo ""
echo "=== Step 2: Image downloads ==="

# --- Caltech-101 ---
echo "[1/10] Caltech-101"
mkdir -p "$DATA_ROOT/caltech101"
cd "$DATA_ROOT/caltech101"
wget -c -q --show-progress -O 101_ObjectCategories.tar.gz \
    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip" 2>/dev/null || \
wget -c -q --show-progress \
    "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
tar -xzf 101_ObjectCategories.tar.gz 2>/dev/null || unzip -q 101_ObjectCategories.tar.gz 2>/dev/null || true
echo "  Done. Expected: $DATA_ROOT/caltech101/101_ObjectCategories/"

# --- DTD ---
echo "[2/10] DTD (Describable Textures)"
mkdir -p "$DATA_ROOT/dtd"
cd "$DATA_ROOT/dtd"
wget -c -q --show-progress "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
tar -xzf dtd-r1.0.1.tar.gz
mv dtd/images images 2>/dev/null || true
echo "  Done. Expected: $DATA_ROOT/dtd/images/"

# --- EuroSAT ---
echo "[3/10] EuroSAT"
mkdir -p "$DATA_ROOT/eurosat"
cd "$DATA_ROOT/eurosat"
wget -c -q --show-progress "https://madm.dfki.de/files/sentinel/EuroSAT.zip" 2>/dev/null || \
wget -c -q --show-progress \
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip"
unzip -q EuroSAT.zip
echo "  Done. Expected: $DATA_ROOT/eurosat/2750/"

# --- FGVC-Aircraft ---
echo "[4/10] FGVC-Aircraft"
mkdir -p "$DATA_ROOT/fgvc"
cd "$DATA_ROOT/fgvc"
wget -c -q --show-progress "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
tar -xzf fgvc-aircraft-2013b.tar.gz
echo "  Done. Expected: $DATA_ROOT/fgvc/data/images/"

# --- Food-101 ---
echo "[5/10] Food-101"
mkdir -p "$DATA_ROOT/food-101"
cd "$DATA_ROOT/food-101"
wget -c -q --show-progress "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
tar -xzf food-101.tar.gz
mv food-101/images images 2>/dev/null || true
echo "  Done. Expected: $DATA_ROOT/food-101/images/"

# --- Oxford Flowers ---
echo "[6/10] Oxford Flowers 102"
mkdir -p "$DATA_ROOT/oxford_flowers"
cd "$DATA_ROOT/oxford_flowers"
wget -c -q --show-progress "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
wget -c -q --show-progress "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
wget -c -q --show-progress "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
tar -xzf 102flowers.tgz
echo "  Done. Expected: $DATA_ROOT/oxford_flowers/jpg/"

# --- Oxford Pets ---
echo "[7/10] Oxford Pets"
mkdir -p "$DATA_ROOT/oxford_pets"
cd "$DATA_ROOT/oxford_pets"
wget -c -q --show-progress "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
wget -c -q --show-progress "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
echo "  Done. Expected: $DATA_ROOT/oxford_pets/images/ and $DATA_ROOT/oxford_pets/annotations/"

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


# --- SUN397 ---
echo "[9/10] SUN397"
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
