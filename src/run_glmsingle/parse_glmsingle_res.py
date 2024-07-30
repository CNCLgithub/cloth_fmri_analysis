import sys, pickle
import numpy as np
from os.path import join as opj
import pandas as pd
sys.path.append('..')
from config import TR, RUNS, VID_PER_RUN, STIMDUR, STIM_DICTS, PROJECT_DIR, SUBJECT_LIST, TASK


outputdir_glmsingle = opj(PROJECT_DIR, 'output', f'glmsingle_{TASK}')


beta_avg_all_subs, block_order_all_subs, bs_order_all_subs, scene_order_all_subs = {}, {}, {}, {}

for sub in SUBJECT_LIST:    
    design_file = opj(PROJECT_DIR, 'data', 'derivatives' , 'fmriprep', f'sub-{sub}', 'func')
    bs_order, scene_order, block_order = [], [], []

    for i in range(1, RUNS+1): 
        cur_design_data = opj(design_file, f'new_sub-{sub}_task-{TASK}CTL_run-{i}_events.tsv')
        df = pd.read_csv(cur_design_data, sep='\t')
        for j in range(1, VID_PER_RUN+1):
            first_idx = df.index[df['trial'] == j].min()
            bs_order.append(df.iloc[first_idx]['bs'])
            scene_order.append(df.iloc[first_idx]['scene'])  
            block_order.append(df.iloc[first_idx]['block'])

    block_order_all_subs[sub] = block_order
    bs_order_all_subs[sub] = bs_order
    scene_order_all_subs[sub] = scene_order
    
    ### load betas
    cur_outputdir_glmsingle = opj(outputdir_glmsingle, f'sub-{sub}', 'TYPED_FITHRF_GLMDENOISE_RR.npy')
    results_glmsingle = np.load(cur_outputdir_glmsingle, allow_pickle=True).item()
    betas = results_glmsingle['betasmd']
    beta_avg_all_subs[sub] = betas

out_file = opj(outputdir_glmsingle, f'beta_data_all_subs_{TASK}.pkl')
with open(out_file, 'wb') as f:
    pickle.dump((block_order_all_subs, bs_order_all_subs, scene_order_all_subs, beta_avg_all_subs), f)

print(f"Dictionaries saved to '{out_file}'")