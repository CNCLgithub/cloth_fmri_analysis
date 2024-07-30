import os, sys
from os.path import join as opj
from os.path import exists
import numpy as np  
import nibabel as nib  
import pandas as pd  
from argparse import ArgumentParser, RawTextHelpFormatter  
from glmsingle.glmsingle import GLM_single  
sys.path.append('..')
from config import TR, RUNS, VID_PER_RUN, STIMDUR, STIM_DICTS, PROJECT_DIR, TASK


parser = ArgumentParser(description='run glmsingle', formatter_class=RawTextHelpFormatter)
parser.add_argument('--sub', type=float, default=7, help='')
opts = parser.parse_args()

sub = int(opts.sub)
sub = '{0:02d}'.format(sub)


datadir_glmsingle = opj(PROJECT_DIR, 'data', 'derivatives' , 'fmriprep', f'sub-{sub}')


### data
data = []
data_dir = opj(datadir_glmsingle, 'func')
for i in range(1, RUNS+1):
    cur_nii_file = opj(data_dir, f'sub-{sub}_task-{TASK}CTL_run-{i}_space-MNI152Lin_desc-preproc_bold.nii.gz')
    cur_nii_data = nib.load(cur_nii_file).get_fdata()
    data.append(cur_nii_data)

xyzt = data[0].shape
xyz = xyzt[:3]


### design
design = []
for i in range(1, RUNS+1):
    design_y_seq = np.zeros((xyzt[-1], len(STIM_DICTS)*RUNS))
    cur_design_data = opj(data_dir, f'new_sub-{sub}_task-{TASK}CTL_run-{i}_events.tsv')
    df = pd.read_csv(cur_design_data, sep='\t')
    
    for j in range(1, VID_PER_RUN+1):        
        first_idx = df.index[df['trial'] == j].min()
        cur_row = first_idx
        cur_scene = df.iloc[first_idx]['scene']
        cur_bs = df.iloc[first_idx]['bs']
        cur_cond = df.iloc[first_idx]['block']
        cur_col = (i-1)*len(STIM_DICTS) + STIM_DICTS[cur_cond]
        design_y_seq[cur_row, cur_col] = 1
        
    design.append(design_y_seq)

    
###
opt = dict()
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]
glmsingle_obj = GLM_single(opt)


outputdir_glmsingle = opj(PROJECT_DIR, 'output', f'glmsingle_{TASK}', f'sub-{sub}')
os.makedirs(outputdir_glmsingle, exist_ok=True)
results_glmsingle = glmsingle_obj.fit(design, data, STIMDUR, TR, outputdir=outputdir_glmsingle)
