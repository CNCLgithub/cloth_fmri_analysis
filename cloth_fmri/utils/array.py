import numpy as np


def norm_dict(data, eps=1e-8):
    all_vals = np.concatenate([
        np.asarray(v).reshape(-1) for v in data.values()
    ])

    vmin = all_vals.min()
    vmax = all_vals.max()
    denom = vmax - vmin + eps

    return {
        k: (np.asarray(v) - vmin) / denom
        for k, v in data.items()
    }


def norm_array(x, eps=1e-8):
    x = np.asarray(x)
    return (x - x.min()) / (x.max() - x.min() + eps)


normalize_data = norm_array
