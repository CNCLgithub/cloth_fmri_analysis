import scipy, random, json, os, sys, pickle, itertools, copy
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
parser.add_argument('--c', type=float, default=0.1, help='0.1|1.0|10.0|100.0')

opts = parser.parse_args()
roi_type = str(opts.roi_type)
c = float(opts.c)
roi_p = 0.05


glmsingle_data = opj(PROJECT_DIR, 'output', f'glmsingle_{TASK}', f'beta_data_all_subs_{TASK}.pkl')
with open(glmsingle_data, 'rb') as f:
    block_order_all_subs, bs_order_all_subs, scene_order_all_subs,  beta_avg_all_subs = pickle.load(f)
    
    
#######
permutations = itertools.permutations(SCENES, 2)
formatted_permutations = ['-'.join(perm) for perm in permutations]
train_acc = {i:None for i in formatted_permutations}


#######
roi_dir = opj(PROJECT_DIR, 'output/glm/towerLoc_space-MNI152Lin')

all_subs_acc = {}
for sub in SUBJECT_LIST: 
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


    #######################
    cur_train_acc = copy.deepcopy(train_acc)
    for cur_scene in SCENES:
        cur_scene_mask = [i == cur_scene for i in scene]
        zmap_train = np.array(zmap)[cur_scene_mask]
        bs_train = np.array(bs)[cur_scene_mask]
        
        ### Train
        n_samples = len(bs_train)
        manual_cv = ManualSplit(range(n_samples), range(n_samples))
        cur_decoder = get_decoder(roi_data, manual_cv, C=c)
        cur_decoder.fit(zmap_train, bs_train)
        
        ### Test
        test_data_mask = [not value for value in cur_scene_mask]
        scene_test = np.array(scene)[test_data_mask]
        bs_test = np.array(bs)[test_data_mask]
        zmap_test = np.array(zmap)[test_data_mask]
        y_pred = cur_decoder.predict(zmap_test)
        acc_pred = y_pred == bs_test
        
        all_test_scenes = set(SCENES) - set([cur_scene])
        for _test_scene in all_test_scenes:
            cur_key = cur_scene + '-' + _test_scene
            cur_test_scene_mask = [i == _test_scene for i in scene_test]
            cur_train_acc[cur_key] = np.mean(acc_pred[cur_test_scene_mask])
    all_subs_acc[sub] = cur_train_acc
    
    
    
#######################
json_file_name = 'test_acc.json-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
out_dir = opj(PROJECT_DIR, 'output', 'rdm', 'train1_test3', TASK, f'{roi_type}-{roi_p}', f'l1-{c}')
os.makedirs(out_dir, exist_ok=True)

out_f = opj(out_dir, json_file_name)
with open(out_f, 'w') as json_file:
    json.dump(all_subs_acc, json_file)
print(f"Saved : {out_f}")
