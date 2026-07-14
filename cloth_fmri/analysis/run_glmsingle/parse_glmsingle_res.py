import pickle
import numpy as np
from os.path import join as opj
import pandas as pd
from cloth_fmri.config.config import CONFIG


beta_avg_all_subs, block_order_all_subs, bs_order_all_subs, scene_order_all_subs = {}, {}, {}, {}

for sub in CONFIG['subjects']:  
    bs_order, scene_order, block_order = [], [], []

    for i in range(1, CONFIG['runs']+1): 
        cur_design_data = opj(CONFIG['fmriprep_root'], f'sub-{sub}', 'func', 
                              f'new_sub-{sub}_task-clothCTL_run-{i}_events.tsv')
        df = pd.read_csv(cur_design_data, sep='\t')
        for j in range(1, CONFIG['videos_per_run']+1):
            first_idx = df.index[df['trial'] == j].min()
            bs_order.append(df.iloc[first_idx]['bs'])
            scene_order.append(df.iloc[first_idx]['scene'])  
            block_order.append(df.iloc[first_idx]['block'])

    block_order_all_subs[sub] = block_order
    bs_order_all_subs[sub] = bs_order
    scene_order_all_subs[sub] = scene_order
    
    ### load betas
    glmsingle_beta_f = opj(CONFIG['glmsingle_root'], f'sub-{sub}', 'TYPED_FITHRF_GLMDENOISE_RR.npy')
    glmsingle_beta = np.load(glmsingle_beta_f, allow_pickle=True).item()
    betas = glmsingle_beta['betasmd']
    beta_avg_all_subs[sub] = betas

    
out_file = opj(CONFIG['glmsingle_root'], 'beta_data_all_subs_typed.pkl')
with open(out_file, 'wb') as f:
    pickle.dump((block_order_all_subs, bs_order_all_subs, scene_order_all_subs, beta_avg_all_subs), f)

print(f"Dictionaries saved to '{out_file}'")