import os
import re
import argparse
import numpy as np
import torch
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.utils import config as config_utils
from src.modeling.factory import build_backbone
from src.data.loader import build_loaders


SCENES = ["wind", "drape", "rotate", "ball"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="path to config.yaml",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="videomae",
        choices=["videomae", "vivit", "vjepa2_hf"],
        help="backbone model",
    )

    parser.add_argument(
        "--pool",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="feature pooling method",
    )

    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        choices=SCENES,
        help="scene to extract: wind | drape | rotate | ball",
    )

    return parser.parse_args()


def clip_name_to_video_name(path):
    """
    xxx_1.npy -> xxx
    xxx_2.npy -> xxx
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    m = re.match(r"^(.*)_\d+$", stem)
    if m is not None:
        return m.group(1)
    return stem


def get_base_dataset_and_indices(dataset):
    """
    Return:
      base_dataset: the underlying dataset that has .samples
      indices: indices into base_dataset corresponding to dataset order
    Works for both raw dataset and torch.utils.data.Subset.
    """
    if isinstance(dataset, Subset):
        base_dataset, base_indices = get_base_dataset_and_indices(dataset.dataset)
        subset_indices = dataset.indices

        if isinstance(subset_indices, torch.Tensor):
            subset_indices = subset_indices.tolist()

        indices = [base_indices[i] for i in subset_indices]
        return base_dataset, indices

    if not hasattr(dataset, "samples"):
        raise ValueError("Dataset must have .samples or be a Subset of such a dataset.")

    indices = list(range(len(dataset)))
    return dataset, indices


def get_sample_paths_for_dataset(dataset):
    base_dataset, indices = get_base_dataset_and_indices(dataset)
    return [base_dataset.samples[i][0] for i in indices]


def infer_scene_from_path(path):
    """
    Try to infer scene from basename first, then parent folder name.
    Expected examples:
      wind_xxx_1.npy
      .../wind_mass_.../clip_1.npy
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parent = os.path.basename(os.path.dirname(path))

    for candidate in (stem, parent):
        for scene in SCENES:
            if candidate == scene or candidate.startswith(scene + "_"):
                return scene

    raise ValueError(f"Cannot infer scene from path: {path}")


def build_scene_loader(loader, scene):
    """
    Filter an existing loader down to one scene only, while preserving loader settings.
    """
    dataset = loader.dataset
    sample_paths = get_sample_paths_for_dataset(dataset)

    keep_indices = [
        i for i, path in enumerate(sample_paths)
        if infer_scene_from_path(path) == scene
    ]

    if len(keep_indices) == 0:
        raise ValueError(f"No samples found for scene '{scene}'.")

    scene_dataset = Subset(dataset, keep_indices)

    scene_loader = DataLoader(
        scene_dataset,
        batch_size=loader.batch_size if loader.batch_size is not None else 1,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
    )

    return scene_loader


@torch.no_grad()
def extract_video_features_mean(loader, backbone, device, pool="mean"):
    """
    Streaming extraction:
    - compute clip features batch by batch
    - aggregate mean feature per video
    - never store all clip features in memory
    """
    backbone.eval()

    sample_paths = get_sample_paths_for_dataset(loader.dataset)

    video_sums = OrderedDict()
    video_counts = OrderedDict()
    video_targets = OrderedDict()

    ptr = 0

    for x, y in tqdm(loader, desc="Extracting and aggregating video features"):
        x = x.to(device, non_blocking=True)

        feat = backbone(x)  # (B, N, D)

        if pool == "cls":
            feat = feat[:, 0]               # (B, D)
        elif pool == "mean":
            feat = feat[:, 1:].mean(dim=1)  # (B, D)
        else:
            raise ValueError(pool)

        feat = feat.cpu().numpy()
        y = y.numpy()

        batch_size = feat.shape[0]
        batch_paths = sample_paths[ptr: ptr + batch_size]
        ptr += batch_size

        for i in range(batch_size):
            video_name = clip_name_to_video_name(batch_paths[i])

            if video_name not in video_sums:
                video_sums[video_name] = feat[i].astype(np.float64, copy=True)
                video_counts[video_name] = 1
                video_targets[video_name] = y[i].copy()
            else:
                video_sums[video_name] += feat[i]
                video_counts[video_name] += 1

                if not np.allclose(video_targets[video_name], y[i]):
                    raise ValueError(
                        f"Target mismatch across clips for video {video_name}: "
                        f"{video_targets[video_name]} vs {y[i]}"
                    )

    video_names = list(video_sums.keys())

    X = np.stack(
        [video_sums[name] / video_counts[name] for name in video_names],
        axis=0
    ).astype(np.float32)

    Y = np.stack([video_targets[name] for name in video_names], axis=0)
    clip_counts = np.array([video_counts[name] for name in video_names], dtype=np.int32)

    return X, Y, np.array(video_names, dtype=object), clip_counts


def save_split_features(loader, backbone, device, pool, out_path):
    X, Y, video_names, clip_counts = extract_video_features_mean(
        loader=loader,
        backbone=backbone,
        device=device,
        pool=pool,
    )

    np.savez(
        out_path,
        X=X,
        Y=Y,
        video_names=video_names,
        clip_counts=clip_counts,
    )

    print(f"Saved: {out_path}")
    print(f"Num videos: {len(video_names)}")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {Y.shape}")


def main():
    args = parse_args()

    cfg = config_utils.load_config(
        args.config,
        overrides=[
            f"model_name={args.model_name}",
            f"head.pool={args.pool}",
        ],
    )

    pool = cfg["head"]["pool"]
    scene = args.scene
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Model:", cfg["model_name"])
    print("Pooling:", pool)
    print("Scene:", scene)

    backbone = build_backbone(cfg).to(device)
    backbone.eval()

    _, _, test_loader, full_train_loader = build_loaders(cfg)

    train_scene_loader = build_scene_loader(full_train_loader, scene)
    test_scene_loader = build_scene_loader(test_loader, scene)

    feature_dir = os.path.join(cfg["dataset"]["feature_dir"], cfg["model_name"])
    os.makedirs(feature_dir, exist_ok=True)

    train_feature_path = os.path.join(
        feature_dir, f"train_video_features_{scene}_{pool}.npz"
    )
    test_feature_path = os.path.join(
        feature_dir, f"test_video_features_{scene}_{pool}.npz"
    )

    if not os.path.exists(train_feature_path):
        print("Extracting train video-level features...")
        save_split_features(
            loader=train_scene_loader,
            backbone=backbone,
            device=device,
            pool=pool,
            out_path=train_feature_path,
        )
    else:
        print("Train features already exist:", train_feature_path)

    if not os.path.exists(test_feature_path):
        print("Extracting test video-level features...")
        save_split_features(
            loader=test_scene_loader,
            backbone=backbone,
            device=device,
            pool=pool,
            out_path=test_feature_path,
        )
    else:
        print("Test features already exist:", test_feature_path)


if __name__ == "__main__":
    main()