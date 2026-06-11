#!/bin/bash -l
#
#SBATCH --job-name=tda_repro
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=10:00:00
#SBATCH --output=output_tda_%j.txt
#SBATCH --export=NONE

# Reproduces the TDA row of Table 1 (CLIP ViT-B/16, cross-dataset benchmark) using the OFFICIAL
# TDA repo (kdiAAA/TDA). Data is symlinked from TCA/data (no re-download). Each dataset is its own
# process -> results/TDA_<ds>.txt. SUN397 omitted (not downloaded).

unset SLURM_EXPORT_ENV
export WANDB_MODE=disabled          # tda_runner.py calls wandb.log() every image, unconditionally

module load python
conda activate TTA

REPO=/home/hpc/rlvl/rlvl168v/MainRepo/TDA
RESULTS=/home/hpc/rlvl/rlvl168v/MainRepo/results
DATA_ROOT=$REPO/data

mkdir -p "$RESULTS"
cd "$REPO"

DATASETS="caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars ucf101"

for d in $DATASETS; do
    out="$RESULTS/TDA_${d}.txt"
    echo "[$(date +%H:%M:%S)] >>> running $d  ->  $out"
    {
        echo "# TDA (positive+negative cache) | ViT-B/16 | dataset=$d | data symlinked from TCA/data"
        echo "# host=$(hostname)  date=$(date)"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 tda_runner.py \
        --config configs \
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

echo "[$(date +%H:%M:%S)] done. TDA results in $RESULTS/ (Table 1: TDA avg 67.53)."
