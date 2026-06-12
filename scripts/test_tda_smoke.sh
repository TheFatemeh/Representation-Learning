#!/bin/bash -l
#
#SBATCH --job-name=tda_smoke
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=00:30:00
#SBATCH --output=output/tda_smoke.out
#SBATCH --export=NONE

# Smoke test: run the official TDA repo on ONE small dataset (dtd, ~1700 imgs) to validate the
# full pipeline (env, wandb-disabled, clip.load from cache, pos/neg caches, symlinked data).
unset SLURM_EXPORT_ENV
export WANDB_MODE=disabled
module load python
conda activate TTA

MAIN="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd $MAIN/TDA
echo "TDA_SMOKE_BEGIN host=$(hostname)"
python3 tda_runner.py --config configs --datasets dtd --backbone ViT-B/16 \
    --data-root $MAIN/TDA/data
echo "TDA_SMOKE_EXIT=$?"
echo "TDA_SMOKE_END  (Table 1 TDA dtd = 47.40)"
