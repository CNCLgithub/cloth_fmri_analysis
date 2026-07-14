# Frozen Transformer + Linear Regression Head (Video)

A video regression pipeline using frozen VideoMAE, ViViT, or V-JEPA backbones with regression head.

## Build the container

```bash
singularity build --fakeroot cloth.sif Singularity.def
```

## Run the pipeline

```bash
# Preprocess raw video frames
sbatch submit_preprocess.sh

# Extract features from the frozen backbone
sbatch submit_preprocess_extract_features.sh

# Train and evaluate the regression head
sbatch submit_train_linear.sh
```

Model and dataset settings are defined in `configs/config.yaml`.
