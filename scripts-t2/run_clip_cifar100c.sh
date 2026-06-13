#!/bin/bash -l
#
#SBATCH --job-name=clip_c100c
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --output=output-t2/clip_c100c_%j.txt
#SBATCH --export=NONE

# Table 2 — CLIP ViT-B/16 zero-shot baseline on CIFAR-100-C, 18-template OpenAI ensemble.
# Pure CLIP: no token pruning (Ours-0.0), no reservoir/TCA. The 18 templates come from the
# dataset (datasets/cifar100c.py) automatically. Contrast / Snow / Brightness, severities 1-5.
# Per-run stdout (accuracy) -> results-t2/CLIP_cifar100c-<corruption>-<sev>.txt

unset SLURM_EXPORT_ENV
module load python
conda activate TTA

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO=$MAIN/TCA
RESULTS=$MAIN/results-t2
DATA_ROOT=$REPO/data

mkdir -p "$RESULTS"
cd "$REPO"

CORRUPTIONS="contrast snow brightness"
SEVERITIES="1 2 3 4 5"

for c in $CORRUPTIONS; do
  for s in $SEVERITIES; do
    ds="cifar100c-${c}-${s}"
    out="$RESULTS/CLIP_${ds}.txt"
    echo "[$(date +%H:%M:%S)] >>> CLIP $ds  ->  $out"
    {
        echo "# CLIP ViT-B/16 zero-shot (18-template ensemble) | dataset=$ds"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 clip_zeroshot.py --backbone ViT-B/16 --datasets $ds --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
        --datasets "$ds" \
        --data-root "$DATA_ROOT" >> "$out" 2>&1
    rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    [ $rc -eq 0 ] && echo "[$(date +%H:%M:%S)] <<< $ds done" \
                  || echo "[$(date +%H:%M:%S)] !!! $ds FAILED (rc=$rc) — see $out"
  done
done

echo "[$(date +%H:%M:%S)] CLIP (18-template) CIFAR-100-C done -> $RESULTS/CLIP_cifar100c-*.txt"
