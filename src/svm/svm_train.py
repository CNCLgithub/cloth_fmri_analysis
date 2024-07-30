import scipy, random, json, os, sys, pickle
import numpy as np
from copy import deepcopy as dp
from os.path import join as opj
import nibabel as nib
import pandas as pd
from datetime import datetime
import argparse 
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
sys.path.append('..')
from config import TR, RUNS, VID_PER_RUN, STIMDUR, STIM_DICTS, PROJECT_DIR, SUBJECT_LIST, TASK, SCENES
from utils import ManualSplit, get_decoder



parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
parser.add_argument('--roi_type', type=str, default="physall", help='v1|physall')
parser.add_argument('--sub', type=int, default=2, help='1-24')
parser.add_argument('--c', type=float, default=0.1, help='0.1|1.0|10.0|100.0')


opts = parser.parse_args()
roi_type = str(opts.roi_type)
c = float(opts.c)
sub = int(opts.sub)
sub = '{0:02d}'.format(sub)


iters = 5
roi_p = 0.05


glmsingle_data = opj(PROJECT_DIR, 'output', f'glmsingle_{TASK}', f'beta_data_all_subs_{TASK}.pkl')
with open(glmsingle_data, 'rb') as f:
    block_order_all_subs, bs_order_all_subs, scene_order_all_subs,  beta_avg_all_subs = pickle.load(f)

    
roi_dir = opj(PROJECT_DIR, 'output/glm/towerLoc_space-MNI152Lin')
roi_file = opj(roi_dir, f'roi_{roi_type}', f'p-{roi_p}', str(sub), f'sub-{sub}_parcel-{roi_type}.nii.gz')
roi_data = nib.load(roi_file)
roi = roi_data.get_fdata()


########################################
bs = np.array(bs_order_all_subs[sub])
scene = np.array(scene_order_all_subs[sub])
betas = np.array(beta_avg_all_subs[sub])
block = np.array(block_order_all_subs[sub])

baseline_mask = block != 'baseline'
bs = list(bs[baseline_mask])
scene = list(scene[baseline_mask])
betas = betas[:,:,:,baseline_mask]

runs_ls = [list(np.zeros(int(len(bs)/RUNS))+i) for i in range(RUNS)]
runs_ls = [item for sublist in runs_ls for item in sublist]
scenerun_ls = [f'{scene[i]}-{runs_ls[i]}' for i in range(len(scene))]

zmap = [betas[:, :, :, i] for i in range(betas.shape[3])]
zmap = [nib.Nifti1Image(i, affine=roi_data.affine) for i in zmap]


########################################
dataset={}
for i_scene in SCENES:
    dataset[i_scene] = {}
    for i_run in range(RUNS):
        dataset[i_scene][i_run] = {}
        cur_train_x, cur_test_x, cur_train_y, cur_test_y, cur_train_scene, cur_test_scene = [], [], [], [], [], []
        for i_idx in range(len(zmap)):
            if runs_ls[i_idx] != i_run and scene[i_idx] != i_scene:
                cur_train_x.append(zmap[i_idx])
                cur_train_y.append(bs[i_idx])
                cur_train_scene.append(f'{scene[i_idx]}-{runs_ls[i_idx]}')
            elif runs_ls[i_idx] == i_run and scene[i_idx] == i_scene:
                cur_test_x.append(zmap[i_idx])
                cur_test_y.append(bs[i_idx])
                cur_test_scene.append(f'{scene[i_idx]}-{runs_ls[i_idx]}')
        dataset[i_scene][i_run]['train_x'] = cur_train_x
        dataset[i_scene][i_run]['train_y'] = cur_train_y
        dataset[i_scene][i_run]['test_x'] = cur_test_x
        dataset[i_scene][i_run]['test_y'] = cur_test_y
        dataset[i_scene][i_run]['train_scene'] = cur_train_scene
        dataset[i_scene][i_run]['test_scene'] = cur_test_scene
        
        
########################################      
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
                tower_decoder = get_decoder(roi_data, manual_cv, C=c)
                tower_decoder.fit(cur_train_x, cur_train_y)
                y_pred = tower_decoder.predict(np.array(cur_train_x)[cur_test_indices])
                y_gt = np.array(cur_train_y)[cur_test_indices]
                cur_acc = np.sum(y_gt == y_pred)/len(cur_test_indices)
                all_acc.append(cur_acc)
                all_run.append(test_run_key)
                all_scene.append(test_scene_key)
            else:
                print(f"!!!!! Skip: {test_scene_key}, {test_run_key}")
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


out_dir = opj(PROJECT_DIR, 'output', 'svm', 'train', TASK, f'{roi_type}-{roi_p}', f'sub-{sub}')
os.makedirs(out_dir, exist_ok=True)
out_f = opj(out_dir, f'l1-{c}.csv')

df.to_csv(out_f, index=False)
print(f"DataFrame saved to {out_f}")