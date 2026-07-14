import argparse
import json
import os
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from os.path import join
from helpers import build_dataset
from cloth_fmri.config.config import CONFIG
from cloth_fmri.utils.svm import ManualSplit, get_decoder
from cloth_fmri.utils.fmri import load_glmsingle_betas, load_roi, prepare_subject_data
from cloth_fmri.utils.paths import get_script_info




def parse_arguments():
    parser = ArgumentParser(description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--sub", type=str, default="02", help="02")
    parser.add_argument("--roi_type", type=str, default="tower", help="v1|tower|loc")
    parser.add_argument("--C", type=float, default=1, help="[0.1, 1, 10, 100]")

    opts = parser.parse_args()

    sub = int(opts.sub)
    sub = "{0:02d}".format(sub)

    roi_type = str(opts.roi_type)
    C = float(opts.C)

    return sub, roi_type, C




def create_validation_sets(dataset):
    for test_scene_key in dataset.keys():
        for test_run_key in dataset[test_scene_key].keys():
            dataset_scenerun = dataset[test_scene_key][test_run_key]

            _train_x = dataset_scenerun["train_x"]
            _train_y = dataset_scenerun["train_y"]
            _train_scenerun = dataset_scenerun["train_scene"]

            _runs = [i.split("-")[-1] for i in _train_scenerun]
            _scenes = [i.split("-")[0] for i in _train_scenerun]

            _unique_scene = list(np.unique(_scenes))
            _unique_runs = list(np.unique(_runs))

            _train_len_all, _val_len_all, _total_len_all = [], [], []
            _all_train_x, _all_train_y, _all_train_scene = [], [], []

            for i_scene in _unique_scene:
                for i_run in _unique_runs:
                    _cur_train_x, _cur_train_y, _cur_train_scene = [], [], []
                    _cur_val_x, _cur_val_y, _cur_val_scene = [], [], []

                    for i_idx in range(len(_scenes)):
                        if _runs[i_idx] != i_run and _scenes[i_idx] != i_scene:
                            _cur_train_x.append(_train_x[i_idx])
                            _cur_train_y.append(_train_y[i_idx])
                            _cur_train_scene.append(_scenes[i_idx])

                        elif _runs[i_idx] == i_run and _scenes[i_idx] == i_scene:
                            _cur_val_x.append(_train_x[i_idx])
                            _cur_val_y.append(_train_y[i_idx])
                            _cur_val_scene.append(_scenes[i_idx])

                    _train_len_all.append(len(_cur_train_x))
                    _val_len_all.append(len(_cur_val_x))

                    _cur_train_x += _cur_val_x
                    _cur_train_y += _cur_val_y
                    _cur_train_scene += _cur_val_scene

                    _total_len_all.append(len(_cur_train_x))
                    _all_train_x.append(_cur_train_x)
                    _all_train_y.append(_cur_train_y)
                    _all_train_scene.append(_cur_train_scene)

            cur_train_dataset = {
                "train_x": _all_train_x,
                "train_y": _all_train_y,
                "train_len_all": _train_len_all,
                "total_len_all": _total_len_all,
                "val_len_all": _val_len_all,
                "train_scene": _all_train_scene,
            }

            dataset[test_scene_key][test_run_key]["train_val_dataset"] = cur_train_dataset

    return dataset


def evaluate_validation_performance(dataset, roi_data, C):
    best_parameters = {}

    for test_scene_key in dataset.keys():
        best_parameters[test_scene_key] = {}

        for test_run_key in dataset[test_scene_key].keys():
            cur_train_dataset = dataset[test_scene_key][test_run_key]["train_val_dataset"]
            acc_C_penalty = {f"{C}-l1": []}

            for i in range(len(cur_train_dataset["train_x"])):
                cur_train_indices = range(0, cur_train_dataset["train_len_all"][i])
                cur_val_indices = range(
                    cur_train_dataset["train_len_all"][i],
                    cur_train_dataset["total_len_all"][i],
                )

                cur_x = cur_train_dataset["train_x"][i]
                cur_y = cur_train_dataset["train_y"][i]

                unique_labels_train = np.unique(np.array(cur_y)[cur_train_indices])
                unique_labels_val = np.unique(np.array(cur_y)[cur_val_indices])

                if len(unique_labels_train) == 2 and len(unique_labels_val) == 2:
                    manual_cv = ManualSplit(cur_train_indices, cur_val_indices)

                    tower_decoder = get_decoder(roi_data, manual_cv, C=C)
                    tower_decoder.fit(cur_x, cur_y)

                    y_pred = tower_decoder.predict(np.array(cur_x)[cur_val_indices])
                    y_gt = np.array(cur_y)[cur_val_indices]

                    cur_acc = np.sum(y_gt == y_pred) / len(cur_val_indices)
                    acc_C_penalty[f"{C}-l1"].append(cur_acc)

            acc_C_penalty = {
                key: np.mean(value)
                for key, value in acc_C_penalty.items()
            }

            best_parameters[test_scene_key][test_run_key] = acc_C_penalty

    return best_parameters


def save_results(best_parameters, data_root, parent_folder_name, script_name, roi_type, sub, C):
    current_datetime_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

#     json_file_path = join(
#         data_root,
#         "derivatives",
#         parent_folder_name,
#         script_name,
#         f"{roi_type}-0.05",
#         f"sub-{sub}",
#         f"l1-{C}",
#     )

    json_file_path = join(data_root, 'derivatives', parent_folder_name, script_name, 
                          f"{roi_type}-smooth={smooth}_c={C}_sub-{sub}")
    
    json_file_name = "val_acc.json-" + current_datetime_str

    os.makedirs(json_file_path, exist_ok=True)

    out_f = join(json_file_path, json_file_name)

    with open(out_f, "w") as json_file:
        json.dump(best_parameters, json_file)

    print(f"Saved : {os.path.abspath(out_f)}")


def main():
    """
    Get gridsearch parameters based on the validation dataset.
    """
    script_name, parent_folder_name = get_script_info(__file__)

    sub, roi_type, C = parse_arguments()

    data_root = CONFIG["data_root"]
    runs = CONFIG["runs"]
    scene_ls = CONFIG["scenes"]

    block_order_all_subs, bs_order_all_subs, scene_order_all_subs, beta_avg_all_subs = load_glmsingle_betas()

    roi_data, roi = load_roi(roi_type, sub)

    bs, scene, betas, runs_ls, zmap = prepare_subject_data(
        sub=sub,
        runs=runs,
        roi_data=roi_data,
        block_order_all_subs=block_order_all_subs,
        bs_order_all_subs=bs_order_all_subs,
        scene_order_all_subs=scene_order_all_subs,
        beta_avg_all_subs=beta_avg_all_subs,
    )

    dataset = build_dataset(
        scene_ls=scene_ls,
        runs=runs,
        zmap=zmap,
        runs_ls=runs_ls,
        scene=scene,
        bs=bs,
    )

    dataset = create_validation_sets(dataset)

    best_parameters = evaluate_validation_performance(
        dataset=dataset,
        roi_data=roi_data,
        C=C,
    )

    save_results(
        best_parameters=best_parameters,
        data_root=data_root,
        parent_folder_name=parent_folder_name,
        script_name=script_name,
        roi_type=roi_type,
        sub=sub,
        C=C,
    )


if __name__ == "__main__":
    main()