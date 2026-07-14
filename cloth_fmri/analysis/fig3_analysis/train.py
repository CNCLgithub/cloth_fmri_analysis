#!/usr/bin/env python
# coding: utf-8

import json
import os
import random
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import join as opj
import numpy as np

from cloth_fmri.config.config import CONFIG
from cloth_fmri.utils.fmri import load_glmsingle_betas, load_roi, prepare_subject_data
from cloth_fmri.utils.svm import ManualSplit
from helpers import get_decoder


warnings.filterwarnings("ignore")


def parse_arguments():
    parser = ArgumentParser(description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--roi_type", type=str, default="tower", help="v1|tower|loc")
    parser.add_argument("--C", type=float, default=1, help="[0.1, 1, 10, 100]")
    parser.add_argument("--leave_runs", type=int, default=1, help="[1, 2]")
    parser.add_argument("--smooth", type=int, default=1, help="[0, 1, 2, 3]")

    opts = parser.parse_args()

    roi_type = str(opts.roi_type)
    C = float(opts.C)
    leave_runs = int(opts.leave_runs)
    smooth = int(opts.smooth)

    return roi_type, C, leave_runs, smooth


def main():
    roi_type, C, leave_runs, smooth = parse_arguments()

    block_order_all_subs, bs_order_all_subs, scene_order_all_subs, beta_avg_all_subs = load_glmsingle_betas()

    tower_acc_all_tmps = {}

    for sub in CONFIG["subjects"]:
        roi_data, _ = load_roi(roi_type=roi_type, sub=sub)

        bs, scene, betas, runs_ls, zmap = prepare_subject_data(
            sub=sub,
            runs=CONFIG["runs"],
            roi_data=roi_data,
            block_order_all_subs=block_order_all_subs,
            bs_order_all_subs=bs_order_all_subs,
            scene_order_all_subs=scene_order_all_subs,
            beta_avg_all_subs=beta_avg_all_subs,
        )

        tower_acc_all_scene = {}

        if leave_runs == 1:
            for cur_scene in CONFIG["scenes"]:
                cur_idx = np.asarray(scene) == cur_scene

                cur_zmap = list(np.asarray(zmap, dtype=object)[cur_idx])
                cur_y = list(np.asarray(bs)[cur_idx])
                cur_leave_one_out_group = list(np.asarray(runs_ls)[cur_idx])

                tower_decoder = get_decoder(roi=roi_data, smoothing_fwhm=smooth, leave_runs=leave_runs, C=C)
                tower_decoder.fit(cur_zmap, cur_y, groups=cur_leave_one_out_group)
                tower_decoder.predict(cur_zmap)

                tower_acc_all_tmp = list(tower_decoder.cv_scores_.values())
                tower_acc = float(np.mean(tower_acc_all_tmp))
                tower_acc_all_scene[cur_scene] = tower_acc

        elif leave_runs == 2:
            for cur_scene in CONFIG["scenes"]:
                tower_acc_all_scene[cur_scene] = []

                cur_idx = np.asarray(scene) == cur_scene

                cur_zmap = list(np.array(zmap)[cur_idx])
                cur_y = list(np.asarray(bs)[cur_idx])
                cur_leave_one_out_group = list(np.asarray(runs_ls)[cur_idx])

                unique_groups = np.unique(cur_leave_one_out_group)

                if len(unique_groups) == 2:
                    for group in unique_groups:
                        cur_train_indices = (np.asarray(cur_leave_one_out_group) == group)
                        cur_test_indices = (np.asarray(cur_leave_one_out_group) != group)
                        cur_train_y = np.asarray(cur_y)[cur_train_indices]

                        if len(np.unique(cur_train_y)) == 2:
                            manual_cv = ManualSplit(cur_train_indices, cur_test_indices)

                            tower_decoder = get_decoder(
                                roi=roi_data,
                                manual_cv=manual_cv,
                                smoothing_fwhm=smooth,
                                leave_runs=leave_runs,
                                C=C,
                            )

                            tower_decoder.fit(cur_zmap, cur_y)

                            test_zmap = np.array(cur_zmap)[cur_test_indices]

                            y_pred = tower_decoder.predict(test_zmap)
                            y_gt = np.asarray(cur_y)[cur_test_indices]

                            cur_acc = float(np.sum(y_gt == y_pred) / len(y_gt))
                            tower_acc_all_scene[cur_scene].append(cur_acc)

        tower_acc_all_tmps[sub] = tower_acc_all_scene

    data = {"tower_acc_all_tmps": tower_acc_all_tmps}

    out_dir = opj(
        CONFIG["output_root"],
        "analysis",
        "fig3",
        f"leave-{leave_runs}-run-out",
        f"C={C}",
        f"smooth={smooth}",
        roi_type,
    )

    os.makedirs(out_dir, exist_ok=True)

    output_file = opj(out_dir, f"output_{random.random()}.json")

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()