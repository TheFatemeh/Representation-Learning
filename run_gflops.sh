#!/bin/bash -l
#
#SBATCH --job-name=tca_gflops
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=output_gflops_%j.txt
#SBATCH --export=NONE

# One-shot GFLOPs (visual encoder, per image) for each method/budget — content-independent,
# so a single value per setting matches Table 1's GFLOPs column. Generic GPU (fast queue).
unset SLURM_EXPORT_ENV
module load python
conda activate TTA
cd /home/hpc/rlvl/rlvl168v/MainRepo/TCA
mkdir -p /home/hpc/rlvl/rlvl168v/MainRepo/results
out=/home/hpc/rlvl/rlvl168v/MainRepo/results/GFLOPs.txt
{
  echo "# GFLOPs (CLIP ViT-B/16 visual encoder, per image)  host=$(hostname)  $(date)"
  echo "# Ours-0.0 = CLIP(R=1.0) | Ours-0.035 = TCA(R=0.9) | EViT-0.1/ToME-0.1 = EViT/ToME(R=0.9) | Ours-0.1 = TCA(R=0.7)"
  for tp in Ours-0.0 Ours-0.035 EViT-0.1 ToME-0.1 Ours-0.1; do
    python3 measure_gflops.py --token_pruning "$tp"
  done
} | tee "$out"
