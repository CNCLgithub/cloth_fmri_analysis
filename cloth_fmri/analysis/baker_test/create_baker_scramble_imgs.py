import os
import random

import cv2
import numpy as np
from os.path import join as opj

from cloth_fmri.config.config import CONFIG


def segment_red_mask(bgr, sat_min=80, val_min=50, k_open=3, k_close=5):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, sat_min, val_min], np.uint8)
    upper1 = np.array([10, 255, 255], np.uint8)

    lower2 = np.array([170, sat_min, val_min], np.uint8)
    upper2 = np.array([180, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    if k_open:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if k_close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return (mask > 0).astype(np.uint8) * 255


def sector_scramble_on_original(
    bgr,
    cloth_mask,
    n_sectors=12,
    center="centroid",
    perm=None,
    seed=0,
    border_mode=cv2.BORDER_CONSTANT,
    annotate=False,
    font_scale=0.6,
    text_color=(0, 255, 0),
    thickness=2,
    angle_offset_deg=0.0,
    center_final=True,
    center_mode="bbox",
):
    H, W = cloth_mask.shape

    if center == "centroid":
        ys, xs = np.where(cloth_mask > 0)
        if xs.size == 0:
            raise ValueError("mask is empty")
        cx, cy = xs.mean(), ys.mean()
    else:
        cx, cy = center

    rng = np.random.RandomState(seed)

    if perm is None:
        perm = np.arange(n_sectors)
        rng.shuffle(perm)
    else:
        perm = np.asarray(perm)
        assert len(perm) == n_sectors and set(perm.tolist()) == set(range(n_sectors)), (
            "perm must be a permutation of 0..n-1"
        )

    yy, xx = np.indices((H, W))

    ang = np.arctan2((cy - yy), (xx - cx))
    ang = (ang + 2 * np.pi) % (2 * np.pi)
    ang = (ang + np.deg2rad(angle_offset_deg)) % (2 * np.pi)

    sector_w = 2 * np.pi / n_sectors
    sector_idx = (ang // sector_w).astype(int)

    out = np.zeros_like(bgr)
    sector_deg = 360.0 / n_sectors

    for k in range(n_sectors):
        tgt = perm[k]
        delta = (tgt - k) * sector_deg

        mask_k = ((cloth_mask > 0) & (sector_idx == k)).astype(np.uint8) * 255

        if mask_k.sum() == 0:
            continue

        seg_k = cv2.bitwise_and(bgr, bgr, mask=mask_k)

        M = cv2.getRotationMatrix2D((cx, cy), delta, 1.0)

        seg_rot = cv2.warpAffine(
            seg_k,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=0,
        )

        msk_rot = cv2.warpAffine(
            mask_k,
            M,
            (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=border_mode,
            borderValue=0,
        )

        if annotate:
            ys2, xs2 = np.where(msk_rot > 0)

            if xs2.size:
                cx2, cy2 = int(xs2.mean()), int(ys2.mean())

                cv2.putText(
                    seg_rot,
                    f"{k}->{tgt}",
                    (cx2, cy2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        m = msk_rot > 0
        out[m] = seg_rot[m]

    if center_final:
        fg_mask = (out.sum(axis=2) > 0).astype(np.uint8)

        ys, xs = np.where(fg_mask > 0)

        if xs.size:
            if center_mode == "centroid":
                cx_new, cy_new = xs.mean(), ys.mean()
            else:
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()

                cx_new = (x0 + x1) / 2.0
                cy_new = (y0 + y1) / 2.0

            cx_tgt, cy_tgt = (W - 1) / 2.0, (H - 1) / 2.0

            dx = cx_tgt - cx_new
            dy = cy_tgt - cy_new

            M_shift = np.array(
                [
                    [1, 0, dx],
                    [0, 1, dy],
                ],
                dtype=np.float32,
            )

            out = cv2.warpAffine(
                out,
                M_shift,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

    return out


def process_all_stimuli():
    input_root_dir = opj(CONFIG["data_root"], "cloth_stimuli")
    output_root_dir = opj(CONFIG["data_root"], "baker_test", "baker_stimuli")

    for ii in range(0, 1): #3):
        for N in [8]: #, 10, 12, 18]:
            for scene in CONFIG["scenes"]:
                scene_root = opj(input_root_dir, scene)

                if not os.path.isdir(scene_root):
                    print(f"Skip: no such dir {scene_root}")
                    continue

                subdirs = sorted(d.name for d in os.scandir(scene_root) if d.is_dir())

                for subdir in subdirs:
                    input_path = opj(scene_root, subdir)

                    first_png = opj(input_path, f"{scene}_cloth_0.png")
                    last_png = opj(input_path, f"{scene}_cloth_199.png")

                    if not (os.path.isfile(first_png) and os.path.isfile(last_png)):
                        print(f"Skip: {input_path} (missing first/last frame)")
                        continue

                    seed = random.randint(0, 1000)

                    for i in range(200):
                        img_path = opj(input_path, f"{scene}_cloth_{i}.png")

                        mask_path = None

                        bgr = cv2.imread(img_path)

                        if bgr is None:
                            raise FileNotFoundError(f"Cannot read {img_path}")

                        if mask_path is not None:
                            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                            if mask is None:
                                raise FileNotFoundError(f"Cannot read {mask_path}")

                            cloth_mask = (mask > 0).astype(np.uint8) * 255

                        else:
                            cloth_mask = segment_red_mask(
                                bgr,
                                sat_min=80,
                                val_min=50,
                                k_open=3,
                                k_close=5,
                            )

                        out = sector_scramble_on_original(
                            bgr,
                            cloth_mask,
                            n_sectors=N,
                            center="centroid",
                            perm=None,
                            seed=seed,
                        )

                        out_path = opj(
                            output_root_dir,
                            f"rendering_baker_scramble_batch{ii}",
                            f"sector={N}",
                            scene,
                            subdir,
                        )

                        os.makedirs(out_path, exist_ok=True)

                        save_path = opj(out_path, f"{scene}_cloth_{i}.png")

                        ok = cv2.imwrite(save_path, out)

                        if not ok:
                            raise IOError(f"Failed to write {save_path}")


def main():
    process_all_stimuli()


if __name__ == "__main__":
    main()