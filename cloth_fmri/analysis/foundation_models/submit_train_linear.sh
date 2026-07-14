#!/bin/bash
#SBATCH --job-name=cloth_linear
#SBATCH --partition=gpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --gres=gpu:1  ##--gpus=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=0-61


# To run: 
# MODEL="vivit" POOL="cls" sbatch submit_train_linear.sh
# MODEL="videomae" POOL="mean" sbatch submit_train_linear.sh
# MODEL="vjepa2_hf" POOL="mean" sbatch submit_train_linear.sh


mkdir -p logs

export HF_HOME=$HOME/hf_cache

echo "Job started on $(hostname) at $(date)"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"

SEED=$((42 + SLURM_ARRAY_TASK_ID))
MODEL=${MODEL:-videomae}
TRAIN_TYPE=${TRAIN_TYPE:-ridge}
ALPHA=${ALPHA:-1.0}
POOL=${POOL:-cls}


srun --cpu-bind=none singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind $HOME/hf_cache:/opt/hf_cache \
  container/cloth.sif \
  bash -c "
    cd /workspace && \
    export PYTHONPATH=\$PYTHONPATH:\$(pwd) && \
    python scripts/train_linear_closed_form.py \
      --config configs/config.yaml \
      --set seed=${SEED} model_name=${MODEL} train.type=${TRAIN_TYPE} train.alpha=${ALPHA} head.pool=${POOL}
  "

echo "Job finished at $(date)"