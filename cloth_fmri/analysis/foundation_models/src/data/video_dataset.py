## src/data/video_dataset.py
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset

from .foldername_parser import parse_folder_name


class ClothFrameFolderDataset(Dataset):
    """
    Dataset that loads preprocessed videos saved as .npy

    Each .npy file shape:
        (T, crop_size, crop_size, 3)
        dtype=uint8

    Preprocessing (resize+crop) is done offline.

    This class only does:
        - frame sampling
        - normalization
        - target loading
    """

    def __init__(
        self,
        root,
        split,
        val_ratio,
        seed,
        targets,
        num_frames,
        sampling_rate,
        resize_short,   # unused but kept for compatibility
        crop_size,      # unused but kept for compatibility
        mean,
        std,
        image_ext=".npy",
        normalize_targets=False,
        frame_indices=None,
        target_mean=None,
        target_std=None,
    ):
        assert split in ("train", "val", "test")

        self.root = root
        self.split = split
        self.targets = targets

        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frame_indices = frame_indices

        self.normalize_targets = normalize_targets
        self.y_mean = None
        self.y_std = None

        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 3)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 3)

        all_samples = []

        for file in sorted(os.listdir(root)):
            if not file.endswith(image_ext):
                continue

            path = os.path.join(root, file)
            name = file[: -len(image_ext)]
            meta = parse_folder_name(name)
            all_samples.append((path, meta))

        if len(all_samples) == 0:
            raise RuntimeError(f"No valid {image_ext} samples found in {root}")

        if split == "test":
            self.samples = all_samples
        else:
            rng = random.Random(seed)
            rng.shuffle(all_samples)
            n_val = int(len(all_samples) * val_ratio)

            if split == "val":
                self.samples = all_samples[:n_val]
            else:
                self.samples = all_samples[n_val:]

        print(f"{split}: {len(self.samples)} samples")
        

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video):
        T = video.shape[0]

        if self.frame_indices is not None:
            idx = np.asarray(self.frame_indices, dtype=int)
        else:
            idx = np.arange(0, T, self.sampling_rate)[: self.num_frames]

        if idx.max() >= T:
            raise IndexError(
                f"Frame index out of range: max idx {idx.max()} but video has {T} frames"
            )

        frames = video[idx]
        return frames

    def _load_video(self, path):
        video = np.load(path, mmap_mode="r")
        frames = self._sample_frames(video)
        return frames

    def __getitem__(self, idx):
        path, meta = self.samples[idx]

        frames = self._load_video(path)
        x = torch.from_numpy(frames).float() / 255.0
        x = (x - self.mean) / self.std
        x = x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        y_raw = torch.tensor(
            [meta[t] for t in self.targets],
            dtype=torch.float32,
        )

        return x, y