#!/bin/bash -l
#
#SBATCH --job-name=tca_clean
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --output=logs/clean_%A_%a.txt
#SBATCH --array=0-15
#SBATCH --export=NONE

# Clean (no-poison) TCA baselines: 4 update rules x M in {1,2,3,5} = 16 runs.
#   - M in {1,3,5} are the experiment's "no contamination" baselines.
#   - M=2 is the paper-validation anchor (Table 4 / eurosat.yaml default):
#       EuroSAT @ M=2 -> FIFO 50.28, Uncertainty 70.20, Similarity 68.48, Diversity 70.43.
# Order/seed match the official repo (global seed 1, stock shuffle).

unset SLURM_EXPORT_ENV
module load python
conda activate TTA

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

RULES=(fifo uncertainty similarity diversity)
MS=(1 2 3 5)
RULE=${RULES[$(( SLURM_ARRAY_TASK_ID / 4 ))]}
M=${MS[$(( SLURM_ARRAY_TASK_ID % 4 ))]}

echo "task=$SLURM_ARRAY_TASK_ID  rule=$RULE  M=$M  host=$(hostname)"
python3 runner.py \
    --backbone ViT-B/16 \
    --token_pruning Ours-0.035 \
    --datasets eurosat \
    --data-root data/ \
    --reservoir_mode clean \
    --update_rule "$RULE" \
    --reservoir_size "$M" \
    --seed 1
