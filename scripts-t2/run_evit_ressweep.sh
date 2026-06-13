#!/bin/bash -l
#
#SBATCH --job-name=evit_ressweep
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=12:00:00
#SBATCH --output=output-t2/evit_ressweep_%j.txt
#SBATCH --export=NONE

# EViT(R=0.9) on CIFAR-100-C at simulated low input resolution(s), to compare its delta-over-CLIP
# against the full-res case. Same EViT setting as Table 1 (EViT-0.1, no reservoir), 18-template.
# Result files: EViT_r<RES>_cifar100c-<c>-<s>.txt
#
# Set RESOLUTIONS to the resolution(s) that matched the paper's CLIP in the CLIP sweep.

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
RESOLUTIONS="32 16 8"                 # <-- narrow to the paper-matching res after the CLIP sweep

for r in $RESOLUTIONS; do
 for c in $CORRUPTIONS; do
  for s in $SEVERITIES; do
    ds="cifar100c-${c}-${s}"
    out="$RESULTS/EViT_r${r}_${ds}.txt"
    echo "[$(date +%H:%M:%S)] >>> EViT res=$r $ds  ->  $out"
    {
        echo "# EViT R=0.9 (EViT-0.1, no reservoir) | effective_res=${r} | dataset=$ds"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 clip_zeroshot.py --backbone ViT-B/16 --token_pruning EViT-0.1 --effective-res $r --datasets $ds --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
        --token_pruning EViT-0.1 \
        --effective-res "$r" \
        --datasets "$ds" \
        --data-root "$DATA_ROOT" >> "$out" 2>&1
    rc=$?
    echo "----------------------------------------------------------------" >> "$out"
    echo "# exit_code=$rc" >> "$out"
    [ $rc -eq 0 ] && echo "[$(date +%H:%M:%S)] <<< res=$r $ds done" \
                  || echo "[$(date +%H:%M:%S)] !!! res=$r $ds FAILED (rc=$rc) — see $out"
  done
 done
done

echo "[$(date +%H:%M:%S)] EViT resolution sweep done -> $RESULTS/EViT_r*_cifar100c-*.txt"
