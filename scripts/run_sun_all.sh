#!/bin/bash -l
#
#SBATCH --job-name=sun_all
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=12:00:00
#SBATCH --output=output/sun_all_%j.txt
#SBATCH --export=NONE

# Runs ALL reproduced methods on SUN397 in one job. Results use the existing per-method
# filenames so they extend the comparison tables:
#   results/CLIP_sun397.txt  TCA_R0.9_sun397.txt  EViT_R0.9_sun397.txt  ToME_R0.9_sun397.txt
#   results/TDA_sun397.txt   CLIPviaTDA_sun397.txt
# Prerequisite: run `bash download_sun.sh` first (images at TCA/data/sun397/SUN397).
# Each method is independent (rc captured) so one failure can't stop the rest.

unset SLURM_EXPORT_ENV
export WANDB_MODE=disabled

module load python
conda activate TTA

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"
TCA=$MAIN/TCA
TDA=$MAIN/TDA
RES=$MAIN/results
mkdir -p "$RES"

# Wire SUN397 into TDA/data (symlinks into TCA/data; safe to re-run)
mkdir -p "$TDA/data/sun397"
ln -sfn "$TCA/data/sun397/SUN397"            "$TDA/data/sun397/SUN397"
ln -sfn "$TCA/splits/split_zhou_SUN397.json" "$TDA/data/sun397/split_zhou_SUN397.json"

run () {  # run <outfile> <workdir> <cmd...>
    local out="$1"; shift; local wd="$1"; shift
    echo "[$(date +%H:%M:%S)] >>> $(basename "$out")"
    {
        echo "# $*"
        echo "# host=$(hostname)  date=$(date)"
        echo "----------------------------------------------------------------"
    } > "$out"
    ( cd "$wd" && "$@" ) >> "$out" 2>&1
    local rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] <<< $(basename "$out") done (rc=0)"
    else
        echo "[$(date +%H:%M:%S)] !!! $(basename "$out") FAILED (rc=$rc)"
    fi
}

# --- CLIP / EViT / ToME / TCA  (TCA repo) ---
run "$RES/CLIP_sun397.txt"      "$TCA" python3 clip_zeroshot.py --backbone ViT-B/16 --datasets sun397 --data-root "$TCA/data"
run "$RES/EViT_R0.9_sun397.txt" "$TCA" python3 clip_zeroshot.py --backbone ViT-B/16 --token_pruning EViT-0.1 --datasets sun397 --data-root "$TCA/data"
run "$RES/ToME_R0.9_sun397.txt" "$TCA" python3 clip_zeroshot.py --backbone ViT-B/16 --token_pruning ToME-0.1 --datasets sun397 --data-root "$TCA/data"
run "$RES/TCA_R0.9_sun397.txt"  "$TCA" python3 runner.py        --backbone ViT-B/16 --token_pruning Ours-0.035 --datasets sun397 --data-root "$TCA/data"

# --- TDA / CLIP-via-TDA  (TDA repo) ---
run "$RES/TDA_sun397.txt"        "$TDA" python3 tda_runner.py --config configs      --datasets sun397 --backbone ViT-B/16 --data-root "$TDA/data"
run "$RES/CLIPviaTDA_sun397.txt" "$TDA" python3 tda_runner.py --config configs_clip --datasets sun397 --backbone ViT-B/16 --data-root "$TDA/data"

echo "[$(date +%H:%M:%S)] SUN397 all-methods done -> $RES/*_sun397.txt"
echo "Paper (SUN397):  CLIP 65.63 | TCA 65.92 | EViT 64.58 | ToME 64.22 | TDA 67.62"
