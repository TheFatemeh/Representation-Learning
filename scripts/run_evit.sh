#!/bin/bash -l
#
#SBATCH --job-name=evit_r09
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=08:00:00
#SBATCH --output=output/evit_%j.txt
#SBATCH --export=NONE

# Reproduces the EViT(R=0.9) row of Table 1: CLIP ViT-B/16 + EViT token pruning (drop 0.1 at
# layers 3/6/9), NO TCA reservoir. Each dataset is its own process -> results/EViT_R0.9_<ds>.txt.
# SUN397 now included (download it first). For faster scheduling, change to --gres=gpu:1 and drop the
# --partition line.

unset SLURM_EXPORT_ENV

module load python
conda activate TTA

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"

REPO=$MAIN/TCA
RESULTS=$MAIN/results
DATA_ROOT=$REPO/data

mkdir -p "$RESULTS"
cd "$REPO"

DATASETS="fgvc caltech101 stanford_cars dtd eurosat oxford_flowers food101 oxford_pets ucf101 sun397"

for d in $DATASETS; do
    out="$RESULTS/EViT_R0.9_${d}.txt"
    echo "[$(date +%H:%M:%S)] >>> running $d  ->  $out"
    {
        echo "# EViT R=0.9 (token_pruning EViT-0.1, no TCA reservoir) | ViT-B/16 | dataset=$d"
        echo "# host=$(hostname)  date=$(date)"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
        --token_pruning EViT-0.1 \
        --datasets "$d" \
        --data-root "$DATA_ROOT" >> "$out" 2>&1
    rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] <<< $d finished (rc=0)"
    else
        echo "[$(date +%H:%M:%S)] !!! $d FAILED (rc=$rc) — see $out"
    fi
done

echo "[$(date +%H:%M:%S)] done. EViT R=0.9 results in $RESULTS/ (Table 1: EViT avg 65.17, GFLOPs 15.41)."
