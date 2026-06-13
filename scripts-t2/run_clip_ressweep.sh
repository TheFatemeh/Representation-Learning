#!/bin/bash -l
#
#SBATCH --job-name=clip_ressweep
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=12:00:00
#SBATCH --output=output-t2/clip_ressweep_%j.txt
#SBATCH --export=NONE

# CLIP ViT-B/16 zero-shot on CIFAR-100-C at several SIMULATED low input resolutions, to find the
# effective resolution that reproduces the paper's much lower CLIP numbers (e.g. contrast: 31.90
# -> 2.69 over sev 1->5). 18-template ensemble, batch size 1 — same as the full-res runs, only
# the input resolution changes. Result files carry the resolution: CLIP_r<RES>_cifar100c-<c>-<s>.txt
# (full-res 224 baseline already exists as CLIP_cifar100c-<c>-<s>.txt from run_clip_cifar100c.sh).

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
RESOLUTIONS="48 32 24 16 8"          # broad search to locate the paper-matching resolution

for r in $RESOLUTIONS; do
 for c in $CORRUPTIONS; do
  for s in $SEVERITIES; do
    ds="cifar100c-${c}-${s}"
    out="$RESULTS/CLIP_r${r}_${ds}.txt"
    echo "[$(date +%H:%M:%S)] >>> CLIP res=$r $ds  ->  $out"
    {
        echo "# CLIP ViT-B/16 zero-shot (18-template) | effective_res=${r} | dataset=$ds"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: python3 clip_zeroshot.py --backbone ViT-B/16 --effective-res $r --datasets $ds --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 clip_zeroshot.py \
        --backbone ViT-B/16 \
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

echo "[$(date +%H:%M:%S)] CLIP resolution sweep done -> $RESULTS/CLIP_r*_cifar100c-*.txt"
