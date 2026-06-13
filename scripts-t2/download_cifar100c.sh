#!/bin/bash
# Download CIFAR-100-C (Table 2). The Zenodo tarball holds all 19 corruptions + labels.npy
# (~2.9 GB). Table 2 only needs Contrast / Snow / Brightness, so we extract just those four
# files (labels.npy is shared across all corruptions).
#
# Run from MainRepo/:   bash scripts-t2/download_cifar100c.sh
#
# After this script:
#   MainRepo/TCA/data/CIFAR-100-C/{contrast,snow,brightness}.npy
#   MainRepo/TCA/data/CIFAR-100-C/labels.npy
#
# To get all 19 corruptions instead, replace the selective `tar` line with: tar -xf "$TAR" -C "$DATA_ROOT"

set -e

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_ROOT="$MAIN/TCA/data"
TAR="$DATA_ROOT/CIFAR-100-C.tar"
URL="https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"

mkdir -p "$DATA_ROOT"

echo "=== Downloading CIFAR-100-C.tar (~2.9 GB) ==="
wget -c --show-progress -O "$TAR" "$URL"

echo "=== Extracting Contrast / Snow / Brightness (+ labels) ==="
tar -xf "$TAR" -C "$DATA_ROOT" \
    CIFAR-100-C/contrast.npy \
    CIFAR-100-C/snow.npy \
    CIFAR-100-C/brightness.npy \
    CIFAR-100-C/labels.npy

echo "  Done. Files in: $DATA_ROOT/CIFAR-100-C/"
ls -lh "$DATA_ROOT/CIFAR-100-C/"
echo "(You can 'rm $TAR' to reclaim ~2.9 GB once you're sure extraction succeeded.)"
