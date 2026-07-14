import random
from collections import Counter
import statistics

import numpy as np
from scipy.stats import spearmanr, ttest_rel


def paired_ttest(x, y):
    t_stat, p_val = ttest_rel(x, y)
    return t_stat, p_val


def calc_p(x, baseline=0):
    x = np.asarray(x)
    return 2 * np.min([
        np.mean(x > baseline),
        np.mean(x < baseline),
    ])


def ci_95(x):
    return np.percentile(x, [2.5, 97.5])


def bootstrap_ci(data, nboot=10000, seed=0):
    data = np.asarray(data)
    rng = np.random.default_rng(seed)

    boots = np.empty(nboot)
    n = len(data)

    for i in range(nboot):
        sample = rng.choice(data, size=n, replace=True)
        boots[i] = np.mean(sample)

    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)

    return lower, upper, boots


def bootstrap_cor(data, nboot=10000, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    n_samples = data.shape[0]
    bootstrap_correlations = []

    for _ in range(nboot):
        sample1_indices = np.random.choice(
            n_samples, n_samples, replace=True
        )
        sample2_indices = np.random.choice(
            n_samples, n_samples, replace=True
        )

        sample1 = data[sample1_indices, :]
        sample2 = data[sample2_indices, :]

        avg_sample1 = np.mean(sample1, axis=0)
        avg_sample2 = np.mean(sample2, axis=0)

        correlation, _ = spearmanr(avg_sample1, avg_sample2)
        bootstrap_correlations.append(correlation)

    mean_r = np.mean(bootstrap_correlations)
    p = calc_p(bootstrap_correlations)

    return mean_r, p, bootstrap_correlations


def get_mode_value(data):
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [
        value
        for value, count in counts.items()
        if count == max_count
    ]

    if len(modes) > 1:
        return statistics.mean(modes)

    return modes[0]


def bootstrap_sample(data):
    return [random.choice(data) for _ in range(len(data))]
