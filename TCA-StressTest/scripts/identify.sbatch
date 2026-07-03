#!/bin/bash -l
#
#SBATCH --job-name=tca_identify
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=01:00:00
#SBATCH --output=logs/identify_%j.txt
#SBATCH --export=NONE

# Stage 2 -- poison identification. One empty-reservoir EuroSAT run: token reduction
# operates but no anchors are stored/used, so the model surfaces its own most-confident
# mistakes. Writes results/identify/poisons.csv + results/identify/poison_seeds/poison_c{c}.pt
# (one per class), which the seeded sweep later injects.
#
# SUBMIT FROM THE REPO ROOT:  sbatch scripts/identify.sbatch
# (the script cd's to $SLURM_SUBMIT_DIR, which must be the TCA-StressTest dir).

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
    --reservoir_mode empty \
    --seed 1
