#!/bin/bash -l
#
#SBATCH --job-name=clip80_c100c
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --output=output-t2/clip80_c100c_%j.txt
#SBATCH --export=NONE

# Table 2 — CLIP ViT-B/16 zero-shot baseline on CIFAR-100-C, 80 hand-crafted ImageNet prompts
# (Radford et al.; the TPT "Ensemble" protocol). --imagenet-ensemble overrides the dataset's
# 18-template ensemble. This is the second CLIP baseline to compare against (see also
# run_clip_cifar100c.sh). Per-run stdout -> results-t2/CLIP80_cifar100c-<corruption>-<sev>.txt

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
    out="$RESULTS/CLIP80_${ds}.txt"
    echo "[$(date +%H:%M:%S)] >>> CLIP80 $ds  ->  $out"
    {
        echo "# CLIP ViT-B/16 zero-shot (80 ImageNet prompts, TPT 'Ensemble') | dataset=$ds"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 clip_zeroshot.py --backbone ViT-B/16 --imagenet-ensemble --datasets $ds --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
        --imagenet-ensemble \
        --datasets "$ds" \
        --data-root "$DATA_ROOT" >> "$out" 2>&1
    rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    [ $rc -eq 0 ] && echo "[$(date +%H:%M:%S)] <<< $ds done" \
                  || echo "[$(date +%H:%M:%S)] !!! $ds FAILED (rc=$rc) — see $out"
  done
done

echo "[$(date +%H:%M:%S)] CLIP (80-prompt) CIFAR-100-C done -> $RESULTS/CLIP80_cifar100c-*.txt"
