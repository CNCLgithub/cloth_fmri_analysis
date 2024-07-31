import random, os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata

def bootstrap_sample(data):
    return [random.choice(data) for _ in range(len(data))]

def list_subdirs(directory):
    subdirs = [os.path.abspath(os.path.join(directory, d)) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirs


def list_files(directory):
    files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def normalize_array(array):
    array = np.array(array)
    mini = np.min(array)
    maxi = np.max(array)
    if mini == maxi:
        return [0] * len(array)
    array_normalized = (array - mini) / (maxi - mini)
    return array_normalized.tolist()


def calc_p (x, baseline=0):
    return 2 * np.min([
        np.mean(np.array(x) > baseline),
        np.mean(np.array(x) < baseline)
    ])


def get_confidence_interval(array, interval=95):
    if len(array) == 0:
        raise ValueError("Array must contain at least one element.")
    
    array = np.sort(array)
    left = (100 - interval) / 2.0
    right = 100 - left
    confidence_interval = np.percentile(array, [left, right])
    
    return confidence_interval

        
def is_convertible_to_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False
    
    
### SVM
class ManualSplit():  
    def __init__(self, train_indices, test_indices):  
        self.train_indices = train_indices  
        self.test_indices = test_indices  
  
    def split(self, X, y=None, groups=None):  
        yield self.train_indices, self.test_indices  

    def get_n_splits(self, X, y, groups=None):  
        return 1  

    
### SVM
def get_decoder(roi, manual_cv, C=1):
    from nilearn.decoding import Decoder
    param_grid = [{"penalty": ["l1"], "dual": [False], "C": [float(C)]}]

    decoder = Decoder(
        estimator="svc",
        smoothing_fwhm=0.0,
        standardize=True,
        mask=roi,
        screening_percentile=100,
        param_grid=param_grid,
        cv=manual_cv)

    return decoder





def split_half_r(my_split_half_data, nboot=10000):
    r_all = []
    for _ in range(nboot):
        np.random.shuffle(my_split_half_data)
        half1 = my_split_half_data[:len(my_split_half_data)//2]
        half2 = my_split_half_data[len(my_split_half_data)//2:]

        mean_half1 = np.mean(half1, axis=0)
        mean_half2 = np.mean(half2, axis=0)
        r = spearmanr(mean_half1, mean_half2)[0]
        r_all.append(r)
    mean_r = round(np.mean(r_all),3)
    p = round(calc_p(r_all),3)
    return mean_r, p



def bootstrap_cor(data, nboot=10000):
    random.seed(9)
    np.random.seed(9)
    
    n_samples = data.shape[0]
    bootstrap_correlations = []

    for _ in range(nboot):
        sample1_indices = np.random.choice(n_samples, n_samples, replace=True)
        sample2_indices = np.random.choice(n_samples, n_samples, replace=True)

        sample1 = data[sample1_indices, :]
        sample2 = data[sample2_indices, :]

        avg_sample1 = np.mean(sample1, axis=0)
        avg_sample2 = np.mean(sample2, axis=0)

        correlation, _ = spearmanr(avg_sample1, avg_sample2)
        bootstrap_correlations.append(correlation)

    mean_r = np.mean(bootstrap_correlations)
    p = calc_p(bootstrap_correlations)
    return mean_r, p


def get_mode_value(data):    
    import statistics
    from collections import Counter
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    
    if len(modes) > 1:
        return statistics.mean(modes)
    else:
        return modes[0]
