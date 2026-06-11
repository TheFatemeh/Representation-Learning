#!/bin/bash -l
#
#SBATCH --job-name=clip_via_tda
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=08:00:00
#SBATCH --output=output_clip_via_tda_%j.txt
#SBATCH --export=NONE

# CLIP zero-shot THROUGH the official TDA repo: tda_runner.py with both caches disabled
# (configs_clip/), so final_logits = clip_logits (pure CLIP, no adaptation). This is the decisive
# check of whether TDA's own released code reproduces the paper's CLIP row. Data symlinked from
# TCA/data. Per-dataset -> results/CLIPviaTDA_<ds>.txt. SUN397 omitted (not downloaded).

unset SLURM_EXPORT_ENV
export WANDB_MODE=disabled

module load python
conda activate TTA

REPO=/home/hpc/rlvl/rlvl168v/MainRepo/TDA
RESULTS=/home/hpc/rlvl/rlvl168v/MainRepo/results
DATA_ROOT=$REPO/data

mkdir -p "$RESULTS"
cd "$REPO"

DATASETS="caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars ucf101"

for d in $DATASETS; do
    out="$RESULTS/CLIPviaTDA_${d}.txt"
    echo "[$(date +%H:%M:%S)] >>> running $d  ->  $out"
    {
        echo "# CLIP via TDA repo (both caches OFF) | ViT-B/16 | dataset=$d | data symlinked from TCA/data"
        echo "# host=$(hostname)  date=$(date)"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 tda_runner.py \
        --config configs_clip \
        --datasets "$d" \
        --backbone ViT-B/16 \
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

echo "[$(date +%H:%M:%S)] done. Compare results/CLIPviaTDA_<ds>.txt  vs  results/CLIP_<ds>.txt (our clip_zeroshot)  vs  paper CLIP."
