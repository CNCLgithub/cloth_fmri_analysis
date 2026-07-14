#!/usr/bin/env python
# coding: utf-8

import copy
import itertools
import json
import os
import random
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import join as opj

import numpy as np

from cloth_fmri.config.config import CONFIG
from cloth_fmri.utils.fmri import (
    load_glmsingle_betas,
    load_roi,
    prepare_subject_data,
)
from cloth_fmri.utils.svm import ManualSplit, get_decoder


warnings.filterwarnings("ignore")


def parse_arguments():
    parser = ArgumentParser(description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--sub", type=str, default="02", help="02")
    parser.add_argument("--roi_type", type=str, default="tower", help="v1|tower|loc")
    parser.add_argument("--C", type=float, default=1, help="[0.1, 1, 10, 100]")

    opts = parser.parse_args()

    sub = "{0:02d}".format(int(opts.sub))
    roi_type = str(opts.roi_type)
    C = float(opts.C)

    return sub, roi_type, C


def main():
    sub, roi_type, C = parse_arguments()

    all_scenes = CONFIG["scenes"]

    (
        block_order_all_subs,
        bs_order_all_subs,
        scene_order_all_subs,
        beta_avg_all_subs,
    ) = load_glmsingle_betas()

    formatted_permutations = [
        "-".join(permutation)
        for permutation in itertools.permutations(all_scenes, 2)
    ]
    train_acc = {permutation: None for permutation in formatted_permutations}

    roi_data, _ = load_roi(roi_type, sub)

    bs, scene, _, _, zmap = prepare_subject_data(
        sub=sub,
        runs=CONFIG["runs"],
        roi_data=roi_data,
        block_order_all_subs=block_order_all_subs,
        bs_order_all_subs=bs_order_all_subs,
        scene_order_all_subs=scene_order_all_subs,
        beta_avg_all_subs=beta_avg_all_subs,
    )

    all_subs_acc = {}
    cur_train_acc = copy.deepcopy(train_acc)

    for cur_scene in all_scenes:
        #######################################################################
        # Train on the current scene
        #######################################################################
        cur_scene_mask = [value == cur_scene for value in scene]

        zmap_train = np.array(zmap)[cur_scene_mask]
        bs_train = np.array(bs)[cur_scene_mask]

        n_samples = len(bs_train)

        manual_cv = ManualSplit(
            range(n_samples),
            range(n_samples),
        )

        cur_decoder = get_decoder(
            roi=roi_data,
            manual_cv=manual_cv,
            penalty="l1",
            C=C,
        )

        cur_decoder.fit(zmap_train, bs_train)

        #######################################################################
        # Test on all other scenes
        #######################################################################
        test_data_mask = [not value for value in cur_scene_mask]

        scene_test = np.array(scene)[test_data_mask]
        bs_test = np.array(bs)[test_data_mask]
        zmap_test = np.array(zmap)[test_data_mask]

        y_pred = cur_decoder.predict(zmap_test)
        acc_pred = y_pred == bs_test

        all_test_scenes = set(all_scenes) - {cur_scene}

        for test_scene in all_test_scenes:
            cur_key = f"{cur_scene}-{test_scene}"
            cur_test_scene_mask = [
                value == test_scene
                for value in scene_test
            ]

            cur_train_acc[cur_key] = np.mean(
                acc_pred[cur_test_scene_mask]
            )

    all_subs_acc[sub] = cur_train_acc

    out_dir = opj(
        CONFIG["output_root"],
        "analysis",
        "fig4",
        roi_type,
        f"C={C}",
    )
    os.makedirs(out_dir, exist_ok=True)

    output_file = opj(
        out_dir,
        f"output_{random.random()}.json",
    )

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(all_subs_acc, json_file)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()