#!/bin/bash -l
#
#SBATCH --job-name=evit_c100c
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --output=output-t2/evit_c100c_%j.txt
#SBATCH --export=NONE

# Table 2 — EViT(R=0.9) on CIFAR-100-C: CLIP ViT-B/16 + EViT token pruning (EViT-0.1), NO TCA
# reservoir. EXACTLY the Table-1 EViT setting. Prompts: the dataset's 18-template ensemble
# (datasets/cifar100c.py). Contrast / Snow / Brightness, severities 1-5.
# Per-run stdout -> results-t2/EViT_cifar100c-<corruption>-<sev>.txt

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
    out="$RESULTS/EViT_${ds}.txt"
    echo "[$(date +%H:%M:%S)] >>> EViT $ds  ->  $out"
    {
        echo "# EViT R=0.9 (token_pruning EViT-0.1, no TCA reservoir) | ViT-B/16 | dataset=$ds"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 clip_zeroshot.py --backbone ViT-B/16 --token_pruning EViT-0.1 --datasets $ds --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
        --token_pruning EViT-0.1 \
        --datasets "$ds" \
        --data-root "$DATA_ROOT" >> "$out" 2>&1
    rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    [ $rc -eq 0 ] && echo "[$(date +%H:%M:%S)] <<< $ds done" \
                  || echo "[$(date +%H:%M:%S)] !!! $ds FAILED (rc=$rc) — see $out"
  done
done

echo "[$(date +%H:%M:%S)] EViT CIFAR-100-C done -> $RESULTS/EViT_cifar100c-*.txt"
