import pickle
from os.path import join
import nibabel as nib
import numpy as np
from cloth_fmri.config.config import CONFIG


def load_glmsingle_betas(glmsingle_data=None):
    
    if glmsingle_data is None:
        glmsingle_data = join(CONFIG["glmsingle_root"], "beta_data_all_subs_typed.pkl")

    with open(glmsingle_data, "rb") as file:
        block_order_all_subs, bs_order_all_subs, scene_order_all_subs, beta_avg_all_subs = pickle.load(file)

    return (block_order_all_subs, bs_order_all_subs, scene_order_all_subs, beta_avg_all_subs)



def get_roi_filename(roi_type, sub):
    """Return the ROI file path."""
    return join(
        CONFIG["roi_root"],
        f"roi_{roi_type}",
        "p-0.05",
        sub,
        f"sub-{sub}_parcel-{roi_type}.nii.gz",
    )


def load_roi(roi_type, sub):
    """Load and return the ROI image and voxel data."""
    roi_file = get_roi_filename(roi_type, sub)

    roi_data = nib.load(roi_file)
    roi = roi_data.get_fdata()

    return roi_data, roi


def prepare_subject_data(
    sub,
    runs,
    roi_data,
    block_order_all_subs,
    bs_order_all_subs,
    scene_order_all_subs,
    beta_avg_all_subs,
):
    bs = np.array(bs_order_all_subs[sub])
    scene = np.array(scene_order_all_subs[sub])
    betas = np.array(beta_avg_all_subs[sub])
    block = np.array(block_order_all_subs[sub])

    baseline_mask = block != "baseline"

    bs = list(bs[baseline_mask])
    scene = list(scene[baseline_mask])
    betas = betas[:, :, :, baseline_mask]

    runs_ls = [
        list(np.zeros(int(len(bs) / runs)) + run)
        for run in range(runs)
    ]
    runs_ls = [
        item
        for run_items in runs_ls
        for item in run_items
    ]

    zmap = [
        betas[:, :, :, i]
        for i in range(betas.shape[3])
    ]
    zmap = [
        nib.Nifti1Image(beta, affine=roi_data.affine)
        for beta in zmap
    ]

    return bs, scene, betas, runs_ls, zmap
