[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

# Causal and statistical object representations in the brain (analysis)

Collection of analysis scripts used in "Computational modeling reveals dissociable causal and statistical object representations in the human brain during spontaneous visual processing".


## 📥 Clone the repository and install it as an editable Python package

Download all required data files by running the following command:

```bash
git clone <repo-url>
cd <repo-name>
python -m pip install -e .
```

## 📥 Download data

Download all required data files by running the following command:

```bash
bash get_data.sh
```


### 🚀 Run analysis and plotting scripts

### Analysis / preprocessing scripts

- [`preprocess`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/preprocess): Scripts for preprocessing raw BOLD signals using fMRIprep.

- [`run_glmsingle`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/cloth_fmri/analysis/run_glmsingle): Run GLMsingle of the cloth run.
  
- [`across_run_split_half_beta_reliability.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/across_run_split_half_beta_reliability.ipynb): Compute model-free odd-even split-half reliability of voxel response patterns across runs.

- [`leave-one-scene-and-one-run-out-svm`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/cloth_fmri/analysis/leave-one-scene-and-one-run-out-svm): Perform the leave-one-scene-and-one-run-out SVM.
  
- [`fig3_analysis`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/cloth_fmri/analysis/fig3_analysis): Get neural decoding accuracy per scenario.
  
- [`fig4_analysis`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/cloth_fmri/analysis/fig4_analysis): Get stiffness similarity matrix for pairs of scenes.
  
- [`create_baker_scramble_imgs.py`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/cloth_fmri/analysis/baker_test/create_baker_scramble_imgs.py): Generate Baker-scrambled control images used in the scrambling analysis.
  
- [`foundation_models`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/cloth_fmri/analysis/foundation_models): A video regression pipeline using frozen VideoMAE, ViViT, or V-JEPA2 backbones with regression head.


### Plotting / Visualization

- [`plot_roi.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/plot_roi.ipynb): Plot subject-specific ROI.
- [`plot_leave-one-scene-and-one-run-out_SVM.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/plot_leave-one-scene-and-one-run-out_SVM.ipynb): Generate figures for Fig. 1D.

- [`plot_svm.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/plot_svm.ipynb): Generate figures for Fig. 3B-E and Supplementary Fig. 6A-C.

- [`plot_rdm.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/plot_rdm.ipynb): Generate figures for Fig. 4B-D and Supplementary Fig. 6D-F.

- [`plot_svm_video_models.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/plot_svm_video_models.ipynb): Generate figures for Fig. 5B and Supplementary Fig. 9A.

- [`plot_rdm_video_models.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/plot_rdm_video_models.ipynb): Generate figures for Fig. 5C and Supplementary Fig. 9B.

- [`Baker_scramble_analysis.ipynb`](https://github.com/CNCLgithub/cloth_fmri_analysis/blob/main/notebooks/Baker_scramble_analysis.ipynb): Generate figures for Baker’s scrambling test in Supplementary Fig. 8.

