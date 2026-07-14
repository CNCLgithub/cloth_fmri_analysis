import numpy as np


def upper_rdm(rdm):
    rdm = np.asarray(rdm)

    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError(f"RDM must be square. Got shape {rdm.shape}.")

    n = len(rdm)
    return rdm[np.triu_indices(n, k=1)]


def upper_to_square_rdm(upper_values, diagonal_value=0.0):
    upper_values = np.asarray(upper_values)
    n = int(np.sqrt(2 * len(upper_values) + 0.25) + 0.5)

    full_rdm = np.full((n, n), diagonal_value, dtype=float)
    indices = np.triu_indices(n, k=1)

    full_rdm[indices] = upper_values
    full_rdm[(indices[1], indices[0])] = upper_values

    return full_rdm


def build_rdm(data_dict, scenes, plot=False):
    rdm = np.zeros((len(scenes), len(scenes)))

    for i in range(len(scenes)):
        for j in range(i + 1, len(scenes)):
            key1 = f"{scenes[i]}-{scenes[j]}"
            key2 = f"{scenes[j]}-{scenes[i]}"
            rdm[i, j] = (data_dict[key1] + data_dict[key2]) / 2.0

    return upper_rdm(rdm)


def vector_to_rdm(rdm_vec, n_conditions, diagonal_value=0.0):
    rdm_vec = np.asarray(rdm_vec)
    expected_len = n_conditions * (n_conditions - 1) // 2

    if len(rdm_vec) != expected_len:
        raise ValueError(
            f"Expected vector length {expected_len}, got {len(rdm_vec)}."
        )

    rdm = np.full(
        (n_conditions, n_conditions),
        diagonal_value,
        dtype=float,
    )
    indices = np.triu_indices(n_conditions, k=1)

    rdm[indices] = rdm_vec
    rdm[(indices[1], indices[0])] = rdm_vec

    return rdm


def normalize_rdm_values(values):
    values = np.asarray(values)
    vmin = values.min()
    vmax = values.max()

    if vmax == vmin:
        raise ValueError("Cannot normalize constant values.")

    return (values - vmin) / (vmax - vmin)


def build_model_rdm_from_stiffness_predictions(model_pred, scenes):
    rdm_model = np.zeros((len(scenes), len(scenes)))

    for i in range(len(scenes)):
        for j in range(i + 1, len(scenes)):
            scene_i_stiff = model_pred[
                f"{scenes[i]}_0.5_2.0"
            ]
            scene_i_soft = model_pred[
                f"{scenes[i]}_0.5_0.0078125"
            ]
            scene_j_stiff = model_pred[
                f"{scenes[j]}_0.5_2.0"
            ]
            scene_j_soft = model_pred[
                f"{scenes[j]}_0.5_0.0078125"
            ]

            mean_scene_i = (
                np.mean(scene_i_stiff)
                + np.mean(scene_i_soft)
            )
            mean_scene_j = (
                np.mean(scene_j_stiff)
                + np.mean(scene_j_soft)
            )

            rdm_model[i, j] = abs(mean_scene_i - mean_scene_j)

    upper_indices = np.triu_indices_from(rdm_model, k=1)
    upper_values = rdm_model[upper_indices]

    normalized_upper = (
        upper_values - upper_values.min()
    ) / (
        upper_values.max() - upper_values.min()
    )

    normalized_rdm = np.zeros_like(rdm_model)
    normalized_rdm[upper_indices] = normalized_upper
    normalized_rdm = normalized_rdm + normalized_rdm.T

    upper_only_rdm = np.triu(normalized_rdm, k=1)
    rdm_vec = 1 - upper_rdm(upper_only_rdm)

    return rdm_vec, upper_only_rdm
