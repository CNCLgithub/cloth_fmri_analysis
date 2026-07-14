import os
import cv2
import numpy as np
from tqdm import tqdm


def resize_and_crop(frames, resize_short, crop_size):
    T, H, W, C = frames.shape

    if min(H, W) != resize_short:
        if H < W:
            new_h = resize_short
            new_w = int(W * resize_short / H)
        else:
            new_w = resize_short
            new_h = int(H * resize_short / W)

        frames = np.stack(
            [cv2.resize(f, (new_w, new_h)) for f in frames],
            axis=0,
        )

    top = (frames.shape[1] - crop_size) // 2
    left = (frames.shape[2] - crop_size) // 2

    frames = frames[
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    return frames


def load_and_preprocess_frames(folder, resize_short, crop_size):
    frame_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".png")]
    )

    if len(frame_files) == 0:
        raise RuntimeError(f"No png frames found in {folder}")

    frames = []

    for fn in frame_files:
        path = os.path.join(folder, fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    frames = np.stack(frames, axis=0)  # (T, H, W, C)
    frames = resize_and_crop(frames, resize_short, crop_size)

    return frames.astype(np.uint8)


def preprocess_split(input_root, output_root, resize_short, crop_size, chunk_size=16):
    os.makedirs(output_root, exist_ok=True)

    folders = sorted(os.listdir(input_root))

    for name in tqdm(folders):
        folder = os.path.join(input_root, name)

        if not os.path.isdir(folder):
            continue

        frames = load_and_preprocess_frames(
            folder,
            resize_short,
            crop_size,
        )

        num_frames = frames.shape[0]
        num_chunks = num_frames // chunk_size  # drop remainder

        if num_chunks == 0:
            print(f"Skip {folder}: fewer than {chunk_size} frames")
            continue

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = frames[start:end]  # shape: (16, H, W, C)

            output_path = os.path.join(output_root, f"{name}_{i+1}.npy")

            if os.path.exists(output_path):
                print(f"Exist: {output_path}")
                continue

            np.save(output_path, chunk)


def main():
    resize_short = 224
    crop_size = 224
    chunk_size = 16   #32 for vivit; 16 for videomae and vjepa2

    preprocess_split(
        "data/raw/train",
        f"data/preprocessed/train_{chunk_size}",
        resize_short,
        crop_size,
        chunk_size,
    )


if __name__ == "__main__":
    main()