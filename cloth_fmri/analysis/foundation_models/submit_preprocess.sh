#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=psych_gpu
#SBATCH --output=%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

export HF_HOME=/opt/hf_cache

echo "Job started on $(hostname) at $(date)"

srun --cpu-bind=none singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind $HOME/hf_cache:/opt/hf_cache \
  cloth.sif \
  bash -c "cd /workspace && export PYTHONPATH=\$PYTHONPATH:\$(pwd) && python -m scripts.preprocess_videos"

echo "Job finished at $(date)"