import os
from argparse import ArgumentParser
from copy import deepcopy as dp
from os.path import join
import numpy as np
import pandas as pd
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



def main():
    script_name, parent_folder_name = get_script_info(__file__)
    
    sub, roi_type, C = parse_arguments()
    
    data_root = CONFIG["data_root"]
    runs = CONFIG["runs"]
    scene_ls = CONFIG["scenes"]
    iters = 5

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


    out_dir = join(data_root, 'derivatives', parent_folder_name, script_name, f"{roi_type}_sub-{sub}")
    os.makedirs(out_dir, exist_ok=True)
    out_f = opj(out_dir, f'l1-{C}.csv')


    ALL_ACC, ALL_RUN, ALL_SCENE, ALL_ITERS = [], [], [], []
    for cur_iter in range(iters):
        all_acc, all_run, all_scene = [], [], []
        for test_scene_key in dataset.keys():
            for test_run_key in dataset[test_scene_key].keys():
                cur_data = dataset[test_scene_key][test_run_key]
                cur_train_x = dp(cur_data['train_x'])
                cur_train_y = dp(cur_data['train_y'])
                cur_test_x = dp(cur_data['test_x'])
                cur_test_y = dp(cur_data['test_y'])

                cur_train_indices = range(0, len(cur_train_x))
                cur_test_indices = range(len(cur_train_x), len(cur_train_x)+len(cur_test_x))
                cur_train_x += cur_test_x
                cur_train_y += cur_test_y

                unique_labels_train = np.unique(np.array(cur_train_y)[cur_train_indices])
                unique_labels_test = np.unique(np.array(cur_train_y)[cur_test_indices])

                if len(unique_labels_train) == 2 and len(unique_labels_test) == 2:
                    manual_cv = ManualSplit(cur_train_indices, cur_test_indices) 
                    tower_decoder = get_decoder(roi_data, manual_cv, C=C)
                    tower_decoder.fit(cur_train_x, cur_train_y)
                    y_pred = tower_decoder.predict(np.array(cur_train_x)[cur_test_indices])
                    y_gt = np.array(cur_train_y)[cur_test_indices]
                    cur_acc = np.sum(y_gt == y_pred)/len(cur_test_indices)
                    all_acc.append(cur_acc)
                    all_run.append(test_run_key)
                    all_scene.append(test_scene_key)

        all_iters = list(np.zeros(len(all_scene)) + cur_iter)

        ALL_ACC += all_acc
        ALL_RUN += all_run
        ALL_SCENE += all_scene
        ALL_ITERS += all_iters


    ########################################    
    ## Save
    data = {
        'ACC': ALL_ACC,
        'run': ALL_RUN,
        'scene': ALL_SCENE,
        'iter': ALL_ITERS
    }
    df = pd.DataFrame(data)
    df.to_csv(out_f, index=False)

    
if __name__ == "__main__":
    main()


