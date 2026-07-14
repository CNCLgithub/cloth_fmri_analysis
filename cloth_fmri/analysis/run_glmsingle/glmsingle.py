import os, sys
from os.path import join as opj
from os.path import exists
import numpy as np  
import nibabel as nib  
import pandas as pd  
from argparse import ArgumentParser, RawTextHelpFormatter  
from glmsingle.glmsingle import GLM_single
from cloth_fmri.config.config import CONFIG
from cloth_fmri.utils.paths import get_script_info


def parse_arguments():
    parser = ArgumentParser(description="glmsingle", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--sub', type=int, default=7, help='')
    
    opts = parser.parse_args()
    sub = "{0:02d}".format(opts.sub)
    
    return sub



def main():
    script_name, parent_folder_name = get_script_info(__file__)
    sub = parse_arguments()
    
    runs = CONFIG["runs"]
    stim_dict = CONFIG['stim_dict']

    ### data
    data = []
    data_dir = opj(CONFIG['fmriprep_root'], f'sub-{sub}', 'func')
    for i in range(1, runs+1):
        cur_nii_file = opj(data_dir, f'sub-{sub}_task-clothCTL_run-{i}_space-MNI152Lin_desc-preproc_bold.nii.gz')
        cur_nii_data = nib.load(cur_nii_file).get_fdata()
        data.append(cur_nii_data)

    xyzt = data[0].shape
    xyz = xyzt[:3]


    ### design
    design = []
    for i in range(1, runs+1):
        design_y_seq = np.zeros((xyzt[-1], len(stim_dict)*runs))
        cur_design_data = opj(data_dir, f'new_sub-{sub}_task-clothCTL_run-{i}_events.tsv')
        df = pd.read_csv(cur_design_data, sep='\t')

        for j in range(1, CONFIG['videos_per_run']+1):        
            first_idx = df.index[df['trial'] == j].min()
            cur_row = first_idx
            cur_scene = df.iloc[first_idx]['scene']
            cur_bs = df.iloc[first_idx]['bs']
            cur_cond = df.iloc[first_idx]['block']
            cur_col = (i-1)*len(stim_dict) + stim_dict[cur_cond]
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

    outputdir_glmsingle = opj(CONFIG['glmsingle_root'], f'sub-{sub}')
    os.makedirs(outputdir_glmsingle, exist_ok=True)
    results_glmsingle = glmsingle_obj.fit(design, data, CONFIG['stim_dur'], CONFIG['TR'], outputdir=outputdir_glmsingle)
    
    
if __name__ == "__main__":
    main()

