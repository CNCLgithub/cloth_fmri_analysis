#!/bin/bash
#SBATCH --job-name=extract_features_by_scene
#SBATCH --partition=gpu
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=260G
#SBATCH --time=24:00:00
#SBATCH --array=0-7      #--array=0-23

set -euo pipefail

export HF_HOME=/opt/hf_cache

echo "Job started on $(hostname) at $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Array task id: ${SLURM_ARRAY_TASK_ID}"

MODELS=(vivit)
POOLS=(cls)
SCENES=(wind drape rotate ball)

TOTAL=$(( ${#MODELS[@]} * ${#POOLS[@]} * ${#SCENES[@]} ))
echo "Total combinations: ${TOTAL}"

TASK_ID=${SLURM_ARRAY_TASK_ID}

MODEL_IDX=$(( TASK_ID / (${#POOLS[@]} * ${#SCENES[@]}) ))
REM=$(( TASK_ID % (${#POOLS[@]} * ${#SCENES[@]}) ))
POOL_IDX=$(( REM / ${#SCENES[@]} ))
SCENE_IDX=$(( REM % ${#SCENES[@]} ))

MODEL_NAME=${MODELS[$MODEL_IDX]}
POOL=${POOLS[$POOL_IDX]}
SCENE=${SCENES[$SCENE_IDX]}

echo "Running combination:"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  POOL=${POOL}"
echo "  SCENE=${SCENE}"

srun --cpu-bind=none singularity exec --nv \
  --bind "$(pwd):/workspace" \
  --bind "$HOME/hf_cache:/opt/hf_cache" \
  cloth.sif \
  bash -c "
    set -euo pipefail
    cd /workspace
    export PYTHONPATH=\$(pwd)
    export HF_HOME=/opt/hf_cache

    python scripts/preprocess_extract_features.py \
      --config configs/config.yaml \
      --model_name ${MODEL_NAME} \
      --pool ${POOL} \
      --scene ${SCENE}
  "

echo "Job finished at $(date)"