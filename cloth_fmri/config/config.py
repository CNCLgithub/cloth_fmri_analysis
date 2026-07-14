from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
DERIVATIVES_ROOT = DATA_ROOT / "derivatives"


CONFIG = {
    "project_name": "cloth_fmri",

    # paths
    "data_root": DATA_ROOT,
    "output_root": PROJECT_ROOT / "outputs",
    "fig_root": PROJECT_ROOT / "figures",
    "derivatives_root": DERIVATIVES_ROOT,
    
    "glmsingle_root": DERIVATIVES_ROOT / "glmsingle",
    "fmriprep_root": DERIVATIVES_ROOT / "fmriprep",
    "roi_root": DERIVATIVES_ROOT / "towerLoc_space-MNI152Lin",
    
    # analysis parameters
    "rois": ["tower", "v1", "loc"],
    "subjects": ["{:02d}".format(n) for n in range(1, 25)],

    "task": "cloth",
    "TR": 0.8,
    "runs": 4,
    "videos_per_run": 20,
    "stim_dur": 0.8 * 8.0,

    "scenes": ["ball", "drape", "rotate", "wind"],
    "stim_dict": {
        "wind-soft": 0,
        "wind-stiff": 1,
        "ball-soft": 2,
        "ball-stiff": 3,
        "drape-soft": 4,
        "drape-stiff": 5,
        "rotate-soft": 6,
        "rotate-stiff": 7,
    }
}