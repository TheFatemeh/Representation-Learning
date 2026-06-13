#!/bin/bash -l
#
#SBATCH --job-name=clip_80prompts
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=08:00:00
#SBATCH --output=output/clip_80prompts_%j.txt
#SBATCH --export=NONE

# CLIP ViT-B/16 zero-shot with the 80 hand-crafted ImageNet prompts (Radford et al.) on ALL
# datasets — the TPT-paper "Ensemble" protocol. The TCA/TDA papers inherited their CLIP row
# from that TPT row (digit-for-digit match), so this run should land on ~the paper CLIP row
# (e.g. EuroSAT ~50.4, Flowers ~67.0, Food ~82.9, Pets ~86.9) and close the CLIP-row mystery.
# Writes to results/CLIP80_<ds>.txt — does NOT touch the CLIP_/EViT_/ToME_/TCA_/TDA_ results.

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
    out="$RESULTS/CLIP80_${d}.txt"
    echo "[$(date +%H:%M:%S)] >>> running $d  ->  $out"
    {
        echo "# CLIP ViT-B/16 zero-shot, 80 ImageNet prompts (TPT 'Ensemble' protocol) | dataset=$d"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 clip_zeroshot.py --backbone ViT-B/16 --imagenet-ensemble --datasets $d --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
        --imagenet-ensemble \
        --datasets "$d" \
        --data-root "$DATA_ROOT" >> "$out" 2>&1
    rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] <<< $d finished (rc=0)"
    else
        echo "[$(date +%H:%M:%S)] !!! $d FAILED (rc=$rc) — see $out — continuing with next dataset"
    fi
done

echo "[$(date +%H:%M:%S)] all datasets attempted. 80-prompt CLIP results in: $RESULTS/CLIP80_*.txt"
