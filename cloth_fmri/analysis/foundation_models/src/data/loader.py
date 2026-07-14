# src/data/loader.py
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from .video_dataset import ClothFrameFolderDataset


def build_loaders(cfg):
    d = cfg["dataset"]
    t = cfg["train"]
    seed = int(cfg["seed"])

    train_root = os.path.join(d["data_dir"], "train")
    test_root = os.path.join(d["data_dir"], "test")

    num_frames = d.get("num_frames")
    if num_frames is None:
        num_frames = cfg["backbone"].get("default_num_frames")

    if num_frames is None:
        raise ValueError(
            "num_frames is missing in dataset config and "
            "no backbone.default_num_frames provided"
        )

    if cfg["backbone"].get("randomize_frame_idx", False):
        np.random.seed(seed)
        frame_indices = np.sort(
            #np.random.choice(d["total_vid_frames"], size=num_frames, replace=False)
            np.random.choice(num_frames, size=num_frames, replace=False)
        )
        frame_indices = frame_indices.astype(int).tolist()
    else:
        frame_indices = None

    full_train_ds = ClothFrameFolderDataset(
        root=train_root,
        split="train",
        val_ratio=0.0,
        seed=seed,
        targets=d["targets"],
        num_frames=num_frames,
        sampling_rate=d["sampling_rate"],
        resize_short=d["resize_short"],
        crop_size=d["crop_size"],
        mean=d["mean"],
        std=d["std"],
        normalize_targets=d.get("normalize_targets", False),
        frame_indices=frame_indices,
    )

    max_train = d.get("max_train_samples", None)
    if max_train is None:
        max_train = len(full_train_ds)
    else:
        max_train = int(max_train)

    if max_train < len(full_train_ds):
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(full_train_ds), generator=g)[:max_train]
        selected_subset = [full_train_ds.samples[i][0] for i in indices.tolist()]
    else:
        indices = torch.arange(len(full_train_ds))
        selected_subset = "all"

    out_dir = cfg["train"]["output_dir"]
    slurm_id = os.environ.get("SLURM_JOB_ID")
    model_name = cfg["model_name"]

    if slurm_id:
        out_dir = os.path.join(out_dir, f"{model_name}_{slurm_id}")

    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, "selected_subset.json")
    with open(save_path, "w") as f:
        json.dump(
            {
                "seed": seed,
                "num_samples": max_train,
                "selected_folders": selected_subset,
                "selected_frame_indices_starting_from_0": frame_indices,
            },
            f,
            indent=2,
        )

    print(f"Saved subset to {save_path}")

    val_ratio = 0
    n_total = len(indices)
    n_val = int(n_total * val_ratio)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_train_ds, val_indices)

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")

    target_mean = full_train_ds.y_mean
    target_std = full_train_ds.y_std

    test_ds = ClothFrameFolderDataset(
        root=test_root,
        split="test",
        val_ratio=0.0,
        seed=seed,
        targets=d["targets"],
        num_frames=num_frames,
        sampling_rate=d["sampling_rate"],
        resize_short=d["resize_short"],
        crop_size=d["crop_size"],
        mean=d["mean"],
        std=d["std"],
        normalize_targets=d.get("normalize_targets", False),
        frame_indices=frame_indices,
        target_mean=target_mean,
        target_std=target_std,
    )

    full_train_eval_ds = ClothFrameFolderDataset(
        root=train_root,
        split="train",
        val_ratio=0.0,
        seed=seed,
        targets=d["targets"],
        num_frames=num_frames,
        sampling_rate=d["sampling_rate"],
        resize_short=d["resize_short"],
        crop_size=d["crop_size"],
        mean=d["mean"],
        std=d["std"],
        normalize_targets=d.get("normalize_targets", False),
        frame_indices=frame_indices,
        target_mean=target_mean,
        target_std=target_std,
    )

    return (
        DataLoader(
            train_ds,
            shuffle=False,
            pin_memory=True,
            batch_size=t["batch_size"],
            num_workers=t.get("num_workers", 4),
        ),
        DataLoader(
            val_ds,
            shuffle=False,
            pin_memory=True,
            batch_size=t["batch_size"],
            num_workers=t.get("num_workers", 4),
        ),
        DataLoader(
            test_ds,
            shuffle=False,
            pin_memory=True,
            batch_size=t["batch_size"],
            num_workers=t.get("num_workers", 4),
        ),
        DataLoader(
            full_train_eval_ds,
            shuffle=False,
            pin_memory=True,
            batch_size=t["batch_size"],
            num_workers=t.get("num_workers", 4),
        ),
    )