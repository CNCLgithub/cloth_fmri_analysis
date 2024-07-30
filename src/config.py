PROJECT_DIR = '/gpfs/milgram/project/yildirim/wb338/github/%github_cloth_fmri_analysis'
SUBJECT_LIST = ['{0:02d}'.format(n) for n in range(1,24)]


TASK = 'cloth'
TR = 0.8
RUNS = 4
VID_PER_RUN = 20
STIMDUR = TR * 8.0
SCENES = ['ball', 'drape', 'rotate', 'wind']
STIM_DICTS = {'wind-soft':0, 'wind-stiff':1, 'ball-soft':2, 'ball-stiff': 3, 'drape-soft':4, 
              'drape-stiff':5, 'rotate-soft': 6, 'rotate-stiff': 7}
