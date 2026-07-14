import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cloth_fmri.utils.rdm import upper_to_square_rdm


def plot_bootstrap_violin(bootstrap_correlations, mean_r, title="Bootstrap Spearman r", ylabel="Spearman correlation"):

    bootstrap_correlations = np.array(bootstrap_correlations)
    ci_low, ci_high = np.percentile(bootstrap_correlations, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(4, 5))

    # Violin plot
    parts = ax.violinplot(bootstrap_correlations, positions=[1],
                          showmeans=False, showmedians=False, showextrema=False)

    # Mean marker
    ax.scatter([1], [mean_r], color="red", marker="o", zorder=3, label="Mean")

    # 95% CI vertical line
    ax.vlines(1, ci_low, ci_high, colors="black", linestyles="-", lw=2, label="95% CI")
    ax.hlines([ci_low, ci_high], 0.9, 1.1, colors="black", lw=1)

    # Formatting
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1])
    ax.set_xticklabels(['Bootstrap r'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    return fig, ax





def plot_scene_bar_with_points(
    normalized,
    means,
    error_bars,
    scenes,
    title,
    ylabel="Normalized accuracy",
):
    """
    Plot scene-wise bar plot with asymmetric error bars and individual points.

    Parameters
    ----------
    normalized : array, shape (n_samples, n_scenes)
        Normalized values for each item/run and scene.
    means : array, shape (n_scenes,)
        Mean value for each scene.
    error_bars : array, shape (2, n_scenes)
        Asymmetric error bars: [lower_err, upper_err].
    scenes : list[str]
        Scene names.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    """
    num_scenes = normalized.shape[1]
    x = np.arange(num_scenes)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        x,
        means,
        yerr=error_bars,
        capsize=5,
        color="skyblue",
        alpha=0.7,
        edgecolor="black",
        label="Mean ± 95% CI",
    )

    for scene_idx in range(num_scenes):
        jitter = np.random.uniform(-0.05, 0.05, size=normalized.shape[0])
        ax.scatter(
            np.full(normalized.shape[0], scene_idx) + jitter,
            normalized[:, scene_idx],
            color="black",
            s=30,
            alpha=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    return fig, ax




def plot_rdm_heatmap_from_upper(
    upper_rdm,
    labels,
    cmap,
    title,
    norm=None,
    annot=True,
    figsize=(8, 6),
):
    """
    Plot an upper-triangular RDM vector as a square heatmap.
    """
    full_rdm = upper_to_square_rdm(upper_rdm)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        full_rdm,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=0.5,
        norm=norm,
        ax=ax,
    )

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)

    ax.set_title(title)
    ax.set_xlabel("Conditions")
    ax.set_ylabel("Conditions")

    plt.tight_layout()

    return fig, ax


def plot_model_rdm_matrix(
    rdm_vec,
    labels,
    title,
    cmap,
    figsize=(10, 8),
):
    """
    Plot model RDM vector as matrix with text labels.
    """
    n = len(labels)
    indices = np.triu_indices(n, k=1)

    rdm = np.zeros((n, n))
    rdm[indices] = rdm_vec

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(rdm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, label="Dissimilarity")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    rdm_max = rdm.max()

    for i in range(len(rdm)):
        for j in range(len(rdm[i])):
            ax.text(
                j,
                i,
                f"{rdm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if rdm[i, j] > rdm_max / 2 else "black",
            )

    plt.tight_layout()

    return fig, ax


def plot_two_distribution_violin(
    data_df,
    cols,
    title,
    ylabel="Values",
    palette=("blue", "orange"),
):
    """
    Plot two bootstrap distributions with mean and 95% CI.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(
        data=data_df[cols],
        ax=ax,
        palette=list(palette),
        inner=None,
        cut=0,
    )

    for i, col in enumerate(cols):
        vals = data_df[col].dropna().values

        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
        mean_val = np.mean(vals)

        ax.vlines(i, ci_low, ci_high, colors="black", linestyles="-", lw=2)
        ax.hlines([ci_low, ci_high], i - 0.1, i + 0.1, colors="black", lw=1)
        ax.scatter(i, mean_val, marker="o", zorder=3, color="black", s=30)

    ax.set_xlabel("Models")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(cols)
    ax.axhline(y=0.0, color="black", linestyle="--")

    plt.tight_layout()

    return fig, ax