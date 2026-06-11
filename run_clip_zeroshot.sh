#!/bin/bash -l
#
#SBATCH --job-name=clip_zeroshot
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=08:00:00
#SBATCH --output=output_clip_zeroshot_%j.txt
#SBATCH --export=NONE

# Reproduces the 'CLIP' row of Table 1 (CLIP ViT-B/16 zero-shot, cross-dataset benchmark).
# Pure CLIP: no token pruning (Ours-0.0), no reservoir/TCA adaptation.
# Each dataset runs as its OWN process (a crash in one can't stop the rest); full stdout+stderr
# (accuracy + GFLOPs) lands in results/CLIP_<ds>.txt. SUN397 omitted (not downloaded).

unset SLURM_EXPORT_ENV

module load python
conda activate TTA

REPO=/home/hpc/rlvl/rlvl168v/MainRepo/TCA
RESULTS=/home/hpc/rlvl/rlvl168v/MainRepo/results
DATA_ROOT=$REPO/data

mkdir -p "$RESULTS"
cd "$REPO"

DATASETS="fgvc caltech101 stanford_cars dtd eurosat oxford_flowers food101 oxford_pets ucf101"
# DATASETS="$DATASETS sun397"   # add once data/sun397/SUN397 exists

for d in $DATASETS; do
    out="$RESULTS/CLIP_${d}.txt"
    echo "[$(date +%H:%M:%S)] >>> running $d  ->  $out"
    {
        echo "# CLIP ViT-B/16 zero-shot baseline | dataset=$d"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 clip_zeroshot.py --backbone ViT-B/16 --datasets $d --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
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

echo "[$(date +%H:%M:%S)] all datasets attempted. per-dataset CLIP results in: $RESULTS/"
