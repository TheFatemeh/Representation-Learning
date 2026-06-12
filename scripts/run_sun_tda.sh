#!/bin/bash -l
#
#SBATCH --job-name=sun_tda
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=04:00:00
#SBATCH --output=output/sun_tda_%j.txt
#SBATCH --export=NONE

# Runs ONLY TDA on SUN397 -> results/TDA_sun397.txt
# Prereq: SUN397 downloaded and un-nested so images are at
#         TCA/data/sun397/SUN397/<letter>/<category>/sun_*.jpg

unset SLURM_EXPORT_ENV
export WANDB_MODE=disabled

module load python
conda activate TTA

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"
TCA=$MAIN/TCA
TDA=$MAIN/TDA
RES=$MAIN/results
mkdir -p "$RES"

# wire SUN397 into TDA/data (symlinks into TCA/data; safe to re-run)
mkdir -p "$TDA/data/sun397"
ln -sfn "$TCA/data/sun397/SUN397"            "$TDA/data/sun397/SUN397"
ln -sfn "$TCA/splits/split_zhou_SUN397.json" "$TDA/data/sun397/split_zhou_SUN397.json"

out="$RES/TDA_sun397.txt"
echo "[$(date +%H:%M:%S)] running TDA on sun397  ->  $out"
{
    echo "# TDA (positive+negative cache) | ViT-B/16 | dataset=sun397 | data symlinked from TCA/data"
    echo "# host=$(hostname)  date=$(date)"
    echo "----------------------------------------------------------------"
} > "$out"
cd "$TDA"
python3 tda_runner.py \
    --config configs \
    --datasets sun397 \
    --backbone ViT-B/16 \
    --data-root "$TDA/data" >> "$out" 2>&1
rc=$?
echo "----------------------------------------------------------------" >> "$out"
echo "# exit_code=$rc" >> "$out"
echo "[$(date +%H:%M:%S)] done (rc=$rc).  Paper TDA SUN397 = 67.62"
