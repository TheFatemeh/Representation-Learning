#!/bin/bash -l
#
#SBATCH --job-name=tca_repro
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --output=output/tca_repro_%j.txt
#SBATCH --export=NONE

# Reproduces Table-1 row TCA(R=0.9), CLIP ViT-B/16, cross-dataset benchmark.
#   --token_pruning Ours-0.035  IS the R=0.9 setting (FLOP-matched to EViT-0.1).
# Each dataset runs as its OWN process: a crash in one is caught and the loop
# continues. Each dataset's full stdout+stderr lands in results/TCA_R0.9_<ds>.txt.

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
    out="$RESULTS/TCA_R0.9_${d}.txt"
    echo "[$(date +%H:%M:%S)] >>> running $d  ->  $out"
    {
        echo "# TCA R=0.9 (Ours-0.035) | backbone ViT-B/16 | dataset=$d"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 runner.py --backbone ViT-B/16 --token_pruning Ours-0.035 --datasets $d --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 runner.py \
        --backbone ViT-B/16 \
        --token_pruning Ours-0.035 \
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

echo "[$(date +%H:%M:%S)] all datasets attempted. per-dataset results in: $RESULTS/"
