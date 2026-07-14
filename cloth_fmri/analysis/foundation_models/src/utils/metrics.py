# src/utils/metrics.py

import numpy as np


def mse(y_true, y_pred):
    """
    Mean squared error per target dimension.
    """
    return ((y_true - y_pred) ** 2).mean(axis=0)

def corr(y_true, y_pred):
    """Pearson correlation per target dimension."""
    corrs = []
    for i in range(y_true.shape[1]):
        corrs.append(np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1])
    return corrs