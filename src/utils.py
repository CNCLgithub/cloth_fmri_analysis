import random, os
import json
import numpy as np


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


def calc_p (array, baseline=0):
    array = np.array(array)
    return 2 * np.min([np.mean(array > baseline), np.mean(array < baseline)])


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