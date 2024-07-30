import sys, random, glob, json
import numpy as np
from collections import defaultdict
from os.path import join as opj
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata

sys.path.append('..')
from config import SCENES



def get_gen_distance(verbose=False):
    random.seed(9)
    np.random.seed(9)


    from model_prediction import gen_stiff_pred, gen_mass_pred
    
    gen_distance = {}
    for scene in SCENES:
        gen_soft = gen_stiff_pred[scene + '_0.5_0.0078125']
        gen_stiff = gen_stiff_pred[scene + '_0.5_2.0']
        random.shuffle(gen_soft)
        random.shuffle(gen_stiff)

        if len(gen_stiff) != len(gen_soft):
            min_length = min(len(gen_stiff), len(gen_soft))
            gen_stiff = gen_stiff[:min_length]
            gen_soft = gen_soft[:min_length]
        gen_distance[scene] = abs(np.array(gen_stiff) - np.array(gen_soft))
        
    if verbose:
        print("GEN Distance:")
        for scene, distance in gen_distance.items():
            print(f"{scene}: {distance}")
        
    return gen_distance


def get_cnn_distance(verbose=False):
    from model_prediction import cnn_stiff_pred, cnn_mass_pred
    
    cnn_distance = {}
    for scene in SCENES:
        cnn_soft = cnn_stiff_pred[scene + '_0.5_0.0078125']
        cnn_stiff = cnn_stiff_pred[scene + '_0.5_2.0']
        cnn_distance[scene] = abs(np.array(cnn_stiff) - np.array(cnn_soft))
        
    if verbose:
        print("\nCNN Distance:")
        for scene, distance in cnn_distance.items():
            print(f"{scene}: {distance}")
    return cnn_distance


def merge_dicts(data_list):
    merged_dict = defaultdict(lambda: defaultdict(dict))

    for data in data_list:
        for key, value in data.items():
            for sub_key, sub_value in value.items():
                for inner_key, inner_value in sub_value.items():
                    if inner_key in merged_dict[key][sub_key]:
                        merged_dict[key][sub_key][inner_key].append(round(inner_value,2))
                    else:
                        merged_dict[key][sub_key][inner_key] = [round(inner_value,2)]
    return {k: dict(v) for k, v in merged_dict.items()}


def find_max_columns(row):
    max_value = row.max()
    return row[row == max_value].index.tolist()



def find_and_load_jsons(data_dir, n=5):
    file_pattern = '*.json*'
    matching_files = glob.glob(f"{data_dir}/{file_pattern}")

    all_data = []
    for file in matching_files[:n]:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    all_data = merge_dicts(all_data)
    return all_data





