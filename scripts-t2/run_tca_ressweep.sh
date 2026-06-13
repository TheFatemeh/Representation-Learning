#!/bin/bash -l
#
#SBATCH --job-name=tca_ressweep
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=output-t2/tca_ressweep_%j.txt
#SBATCH --export=NONE

# TCA(R=0.9) on CIFAR-100-C at simulated low input resolution(s) — the key test: does TCA's gain
# over CLIP only appear once the CLIP baseline is degraded (low-res)? Same TCA setting as Table 1
# (Ours-0.035), EuroSAT-template config, 18-template prompts.
#
# Uses K=2 merge centers (paper's value, supp Tab. 10) via TCA_MERGE_K=2 — the only run where we
# override the code's default K=4. Result files: TCA_R0.9_r<RES>_cifar100c-<c>-<s>.txt
#
# TCA is batch_size=1 with the reservoir: each resolution = 3 x 5 x 10000 = 150k forward passes.
# Keep RESOLUTIONS small; set it to the res that matched the paper's CLIP in the CLIP sweep.

unset SLURM_EXPORT_ENV
module load python
conda activate TTA

export TCA_MERGE_K=2                   # paper K=2 (code default is 4)

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO=$MAIN/TCA
RESULTS=$MAIN/results-t2
DATA_ROOT=$REPO/data

mkdir -p "$RESULTS"
cd "$REPO"

CORRUPTIONS="contrast snow brightness"
SEVERITIES="1 2 3 4 5"
RESOLUTIONS="16"                       # <-- set to the paper-matching res (one value = ~1 full TCA run)

for r in $RESOLUTIONS; do
 for c in $CORRUPTIONS; do
  for s in $SEVERITIES; do
    ds="cifar100c-${c}-${s}"
    out="$RESULTS/TCA_R0.9_r${r}_${ds}.txt"
    echo "[$(date +%H:%M:%S)] >>> TCA res=$r K=$TCA_MERGE_K $ds  ->  $out"
    {
        echo "# TCA R=0.9 (Ours-0.035) | effective_res=${r} | merge_K=${TCA_MERGE_K} | dataset=$ds"
        echo "# host=$(hostname)  date=$(date)"
        echo "# cmd: TCA_MERGE_K=$TCA_MERGE_K python3 runner.py --backbone ViT-B/16 --token_pruning Ours-0.035 --effective-res $r --datasets $ds --data-root $DATA_ROOT"
        echo "----------------------------------------------------------------"
    } > "$out"
    python3 runner.py \
        --backbone ViT-B/16 \
        --token_pruning Ours-0.035 \
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

echo "[$(date +%H:%M:%S)] TCA resolution sweep done -> $RESULTS/TCA_R0.9_r*_cifar100c-*.txt"
