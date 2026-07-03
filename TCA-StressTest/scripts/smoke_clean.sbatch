#!/bin/bash -l
#
#SBATCH --job-name=tca_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_clean_%j.txt
#SBATCH --export=NONE

# Smoke test: instrumented TCA, clean mode, default M (=2 from eurosat.yaml),
# diversity rule. Must reproduce the paper's TCA R=0.9 EuroSAT = 70.43%.

unset SLURM_EXPORT_ENV
module load python
conda activate TTA

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

python3 runner.py \
    --backbone ViT-B/16 \
    --token_pruning Ours-0.035 \
    --datasets eurosat \
    --data-root data/ \
    --reservoir_mode clean \
    --update_rule diversity \
    --seed 1
