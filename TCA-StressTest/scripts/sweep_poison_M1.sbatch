#!/bin/bash -l
#
#SBATCH --job-name=tca_pois_M1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --output=logs/poison_M1_%A_%a.txt
#SBATCH --array=0-39
#SBATCH --export=NONE

# Stage 3 -- poisoned sweep, M=1. 4 rules x 10 target classes = 40 runs.
# Each task seeds buffer R_c with results/identify/poison_seeds/poison_c{c}.pt at t=0.
# SUBMIT FROM THE REPO ROOT:  sbatch scripts/sweep_poison_M1.sbatch

unset SLURM_EXPORT_ENV
module load python
conda activate TTA

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

M=1
RULES=(fifo uncertainty similarity diversity)
RULE=${RULES[$(( SLURM_ARRAY_TASK_ID / 10 ))]}
C=$(( SLURM_ARRAY_TASK_ID % 10 ))

echo "task=$SLURM_ARRAY_TASK_ID  rule=$RULE  M=$M  target=$C  host=$(hostname)"
python3 runner.py \
    --backbone ViT-B/16 \
    --token_pruning Ours-0.035 \
    --datasets eurosat \
    --data-root data/ \
    --reservoir_mode seeded \
    --update_rule "$RULE" \
    --reservoir_size $M \
    --target_class $C \
    --poison_path results/identify/poison_seeds/poison_c${C}.pt \
    --seed 1
