#!/bin/bash -l
#
#SBATCH --job-name=tca_c100c
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=output-t2/tca_c100c_%j.txt
#SBATCH --export=NONE

# Table 2 — TCA(R=0.9) on CIFAR-100-C ("Ours" column). --token_pruning Ours-0.035 IS R=0.9,
# EXACTLY the Table-1 TCA setting. TCA hyper-parameters come from configs/cifar100c.yaml
# (a copy of EuroSAT's — the paper gives none for CIFAR-100-C; see summary/Table2.md).
# Prompts: the dataset's 18-template ensemble. Contrast / Snow / Brightness, severities 1-5.
#
# NOTE: TCA runs batch_size=1 with the reservoir -> 3 x 5 x 10 000 = 150 000 forward passes.
# This is the long pole; --time is set to 24h accordingly.
# Per-run stdout -> results-t2/TCA_R0.9_cifar100c-<corruption>-<sev>.txt

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
    out="$RESULTS/TCA_R0.9_${ds}.txt"
    echo "[$(date +%H:%M:%S)] >>> TCA $ds  ->  $out"
    {
        echo "# TCA R=0.9 (Ours-0.035) | backbone ViT-B/16 | dataset=$ds"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 runner.py --backbone ViT-B/16 --token_pruning Ours-0.035 --datasets $ds --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 runner.py \
        --backbone ViT-B/16 \
        --token_pruning Ours-0.035 \
        --datasets "$ds" \
        --data-root "$DATA_ROOT" >> "$out" 2>&1
    rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    [ $rc -eq 0 ] && echo "[$(date +%H:%M:%S)] <<< $ds done" \
                  || echo "[$(date +%H:%M:%S)] !!! $ds FAILED (rc=$rc) — see $out"
  done
done

echo "[$(date +%H:%M:%S)] TCA CIFAR-100-C done -> $RESULTS/TCA_R0.9_cifar100c-*.txt"
