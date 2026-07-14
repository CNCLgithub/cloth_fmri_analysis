import os
import argparse
import numpy as np
import torch

from src.utils import config as config_utils
from src.utils.metrics import mse, corr
from src.modeling.decoder_factory import build_decoder
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Overrides like seed=123 model_name=videomae train.type=ridge dataset.n_train=3000",
    )
    return parser.parse_args()


def build_output_dir(cfg):
    base_out_dir = cfg["train"]["output_dir"]
    os.makedirs(base_out_dir, exist_ok=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    model_name = cfg["model_name"]

    targets = cfg["dataset"]["targets"]
    targets_str = "-".join(targets)

    if not slurm_job_id:
        slurm_job_id = '000000'
        print(f"@@@@ Warning: No SLURM info, default to 000000")
    
    out_dir = os.path.join(base_out_dir, 
                            f'{cfg["train"]["type"]}_{cfg["head"]["pool"]}_{cfg["train"]["alpha"]}',
                            f'{targets_str}_ntrain={cfg["dataset"]["n_train"]}',
                            f"{model_name}_{slurm_job_id}")

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    args = parse_args()
    cfg = config_utils.load_config(args.config, overrides=args.set)
    set_seed(int(cfg["seed"]))

    out_dir = build_output_dir(cfg)

    config_utils.save_config(
        cfg,
        os.path.join(out_dir, "resolved_config.yaml"),
    )

    # ----------------------------
    # Load pre-extracted features
    # ----------------------------
    feature_dir = os.path.join(cfg["dataset"]["feature_dir"], cfg["model_name"])
    pool = cfg["head"]["pool"]


    ################################################################
    SCENES = ["wind", "drape", "rotate", "ball"]

    X_train_list = []
    Y_train_list = []
    file_names_list = []

    X_test_list = []
    Y_test_list = []
    test_file_names_list = []

    for scene in SCENES:

        train_path = os.path.join(
            feature_dir,
            f"train_video_features_{scene}_{pool}.npz"
        )

        test_path = os.path.join(
            feature_dir,
            f"test_video_features_{scene}_{pool}.npz"
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(train_path)

        if not os.path.exists(test_path):
            raise FileNotFoundError(test_path)

        train_data = np.load(train_path, allow_pickle=True)
        test_data = np.load(test_path, allow_pickle=True)

        X_train_list.append(train_data["X"])
        Y_train_list.append(train_data["Y"])
        file_names_list.append(train_data["video_names"])

        X_test_list.append(test_data["X"])
        Y_test_list.append(test_data["Y"])
        test_file_names_list.append(test_data["video_names"])

    # concatenate scenes
    X_full = np.concatenate(X_train_list, axis=0)
    Y_full = np.concatenate(Y_train_list, axis=0)
    file_names_full = np.concatenate(file_names_list, axis=0)

    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)
    test_file_names = np.concatenate(test_file_names_list, axis=0)

    print("Loaded scenes:", SCENES)
    print("Total training videos:", len(X_full))
    print("Total test videos:", len(X_test))
    ################################################################


    targets = cfg["dataset"]["targets"]
    if "mass" not in targets:
        Y_full = Y_full[:, 0:1]
        Y_test = Y_test[:, 0:1]

    # ----------------------------
    # Randomly sample n_train examples from full training set
    # ----------------------------
    seed = int(cfg["seed"])
    n_train = int(cfg["dataset"].get("n_train"))

    if n_train > len(X_full):
        raise ValueError(
            f"Requested n_train={n_train}, but only {len(X_full)} training samples are available."
        )

    g = torch.Generator()
    g.manual_seed(seed)
    train_idx = torch.randint(0, len(X_full), (n_train,), generator=g).numpy()

    X_train = X_full[train_idx]
    Y_train = Y_full[train_idx]
    train_file_names = file_names_full[train_idx]

    print(f"Full training pool size: {len(X_full)}")
    print(f"Randomly selected training size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # ----------------------------
    # Train decoder
    # ----------------------------
    decoder = build_decoder(cfg)

    print(f"Training decoder: {decoder.__class__.__name__}")
    decoder.fit(X_train, Y_train)

    train_pred = decoder.predict(X_train)
    test_pred = decoder.predict(X_test)

    train_mse = mse(Y_train, train_pred)
    test_mse = mse(Y_test, test_pred)

    train_corr = corr(Y_train, train_pred)
    test_corr = corr(Y_test, test_pred)


    # ----------------------------
    # Save results
    # ----------------------------
    decoder_type = cfg["train"]["type"]
    save_path = os.path.join(out_dir, f"{decoder_type}_decoder_results.npz")

    data_to_save = {
        "X_train": X_train,
        "Y_train": Y_train,
        "train_pred": train_pred,
        "X_test": X_test,
        "Y_test": Y_test,
        "test_pred": test_pred,
        "train_mse": np.array(train_mse),
        "test_mse": np.array(test_mse),
        "train_corr": np.array(train_corr),
        "test_corr": np.array(test_corr),
        "train_file_names": np.array(train_file_names, dtype=object),
        "test_file_names": np.array(test_file_names, dtype=object),
        "train_idx": train_idx,
        "n_train": np.array(n_train),
    }

    if hasattr(decoder, "get_params"):
        data_to_save.update(decoder.get_params())

    np.savez(save_path, **data_to_save)
    print(f"Saved decoder results to {save_path}")


if __name__ == "__main__":
    main()