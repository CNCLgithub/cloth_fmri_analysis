import scipy, random, json, os, sys, pickle
import numpy as np
from os.path import join as opj
import nibabel as nib
from datetime import datetime
import argparse 
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
sys.path.append('..')
from config import TR, RUNS, VID_PER_RUN, STIMDUR, STIM_DICTS, PROJECT_DIR, SUBJECT_LIST, TASK, SCENES
from utils import ManualSplit, get_decoder


parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
parser.add_argument('--roi_type', type=str, default="physall", help='v1|physall')
parser.add_argument('--c', type=str, default='1.0', help='0.1|1.0|10.0|100.0')
parser.add_argument('--sub', type=int, default=2, help='1-24')


opts = parser.parse_args()
roi_type = str(opts.roi_type)
c = str(opts.c)
sub = int(opts.sub)
sub = '{0:02d}'.format(sub)


roi_p = 0.05
c_penalty = f'{c}-l1'
penalty_c = f'l1-{c}'

glmsingle_data = opj(PROJECT_DIR, 'output', f'glmsingle_{TASK}', f'beta_data_all_subs_{TASK}.pkl')
with open(glmsingle_data, 'rb') as f:
    block_order_all_subs, bs_order_all_subs, scene_order_all_subs,  beta_avg_all_subs = pickle.load(f)
    
    
roi_dir = opj(PROJECT_DIR, 'output/glm/towerLoc_space-MNI152Lin')
roi_file = opj(roi_dir, f'roi_{roi_type}', f'p-{roi_p}', str(sub), f'sub-{sub}_parcel-{roi_type}.nii.gz')
roi_data = nib.load(roi_file)
roi = roi_data.get_fdata()


##########
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



##########
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
        
        
########## Create for validation
for test_scene_key in dataset.keys():
    for test_run_key in dataset[test_scene_key].keys():
        dataset_scenerun = dataset[test_scene_key][test_run_key]
        _train_x = dataset_scenerun['train_x']
        _train_y = dataset_scenerun['train_y']
        _train_scenerun = dataset_scenerun['train_scene']

        _runs = [i.split('-')[-1] for i in _train_scenerun]
        _scenes = [i.split('-')[0] for i in _train_scenerun]
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

        cur_train_dataset = {}
        cur_train_dataset['train_x'] = _all_train_x
        cur_train_dataset['train_y'] = _all_train_y
        cur_train_dataset['train_len_all'] = _train_len_all
        cur_train_dataset['total_len_all'] = _total_len_all
        cur_train_dataset['val_len_all'] = _val_len_all
        cur_train_dataset['train_scene'] = _all_train_scene
        dataset[test_scene_key][test_run_key]['train_val_dataset'] = cur_train_dataset
        
        
##########
best_parameters = {}
for test_scene_key in dataset.keys():
    best_parameters[test_scene_key] = {}
    for test_run_key in dataset[test_scene_key].keys():
        cur_train_dataset = dataset[test_scene_key][test_run_key]['train_val_dataset']
        acc_c_penalty = {c_penalty: []}
        
        for i in range(len(cur_train_dataset['train_x'])):
            cur_train_indices = range(0, cur_train_dataset['train_len_all'][i])
            cur_val_indices = range(cur_train_dataset['train_len_all'][i], cur_train_dataset['total_len_all'][i])
            cur_x = cur_train_dataset['train_x'][i]
            cur_y = cur_train_dataset['train_y'][i]
            unique_labels_train = np.unique(np.array(cur_y)[cur_train_indices])
            unique_labels_val = np.unique(np.array(cur_y)[cur_val_indices])
            if len(unique_labels_train) == 2 and len(unique_labels_val) == 2:
                manual_cv = ManualSplit(cur_train_indices, cur_val_indices) 
                tower_decoder = get_decoder(roi_data, manual_cv, C=float(c))
                tower_decoder.fit(cur_x, cur_y)
                y_pred = tower_decoder.predict(np.array(cur_x)[cur_val_indices])
                y_gt = np.array(cur_y)[cur_val_indices]
                cur_acc = np.sum(y_gt == y_pred)/len(cur_val_indices)
                acc_c_penalty[c_penalty].append(cur_acc)
        
        acc_c_penalty = {key:np.mean(value) for key, value in acc_c_penalty.items()}
        best_parameters[test_scene_key][test_run_key] = acc_c_penalty
        
        
##########        
current_datetime_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
json_file_path = opj(PROJECT_DIR, 'output', 'svm', 'validation_acc_grid_search', TASK, 
                     f'{roi_type}-{roi_p}', f'sub-{sub}', penalty_c)
json_file_name = 'val_acc.json-' + current_datetime_str
os.makedirs(json_file_path, exist_ok=True)
out_f = opj(json_file_path, json_file_name)
with open(out_f, 'w') as json_file:
    json.dump(best_parameters, json_file)
print(f"Saved : {os.path.abspath(out_f)}")
