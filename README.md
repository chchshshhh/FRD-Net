# AAAI 2026 – Amplifying Discrepancies: Exploiting Macro and Micro Inconsistencies for Image Manipulation Localization


This repository contains the code and configuration files used in our AAAI 2026 paper:

> **“Amplifying Discrepancies: Exploiting Macro and Micro Inconsistencies for Image Manipulation Localization”**



Our method, **FRD-Net**, is implemented on top of the **IMDL-BenCo** codebase, a comprehensive benchmark and framework for image manipulation detection and localization.

- **Paper:** [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40844) / [DOI](https://doi.org/10.1609/aaai.v40i42.40844)
- **Supplementary material:** [`docs/AAAI_2026_FRD_Net_sup.pdf`](docs/AAAI_2026_FRD_Net_sup.pdf)


![Pipeline of FRD-Net](images/framework.png)
![Pipeline of FRD-Net](images/Grad-CAM.png)
---
### Pre-trained Weights

You can download the pre-trained **FRD-Net** weights from Google Drive:

- [FRD-Net weights and Grad-CAM result](https://drive.google.com/drive/folders/1ty47irsz7FOTjzVNzb6lR3u8wM9eSqaR?usp=sharing)

## 1. Repository Structure

At the top level:

- `IMDLBenCo/`  
  Core benchmark code (datasets, transforms, evaluation metrics, training utilities, model registry, etc.)

- `IMDLBenCo/modules/backbones/FRDNet.py`  
  Implementation of **FRD-Net**, our backbone that exploits macro and micro inconsistencies.

- `docs/`  
  Supplementary material for the AAAI 2026 paper.

- `runs/`  
  Shell scripts for running the main experiments:
  - `demo_train_FRDNet.sh` – example script for training FRD-Net.
  - `demo_test_FRDNet.sh` – example script for evaluating a trained model on multiple benchmarks.

- `weights/`  
  Placeholder directory for saving / placing trained checkpoints used for evaluation.

- `balanced_dataset.json`, `balanced_dataset2.json`  
  Training dataset compositions (lists of IMDL-BenCo JSON datasets for supervised training).

- `test_datasets.json`  
  List of test sets (e.g. Columbia, COVER, CASIA, CocoGlide, NIST16) used to reproduce the main evaluation in the paper.

- `test_robust.py`, `test_rob.json`  
  Scripts and configuration for robustness evaluation under various perturbations.

- `train_val.py`, `train_valpar.py`  
  Training entry points built on IMDL-BenCo’s training framework.

- `test.py`  
  Evaluation entry point for multi-dataset testing using `test_datasets.json`.

- `requirements/`  
  Environment configuration:
  - `conda_requirements.txt`
  - `pip_requirement.txt`


---

## 2. Environment Setup

We recommend creating a **conda environment** (Python 3.8–3.10) and using a recent CUDA-enabled PyTorch.

```bash
# 1) Create and activate env (example)
conda create -n frdnet python=3.10
conda activate frdnet

# 2) Install dependencies (adjust according to your system)
# Option A: using conda_requirements.txt as reference
#   You can manually install key libraries from there.
# Option B: install pip requirements directly
pip install -r requirements/pip_requirement.txt
```

## 3. Data Preparation

IMDL-BenCo organizes datasets via JSON index files.
In this repository, the provided JSONs (e.g. `balanced_dataset.json`, `test_datasets.json`) contain **absolute paths** to our local datasets (e.g. `/disk/csh/IMDLBenCo/...`).

To run on your machine:

1. **Prepare datasets** used in the paper (e.g. CASIA v1/v2, IMD2020, FantasticReality, etc.) following the IMDL-BenCo format (JsonDataset / ManiDataset).
2. **Update the JSON files** so that each path points to your local directory structure.

   * `balanced_dataset.json`: training sets.(CAT-Net Split a portion of the training set as the validation set.)
   * `balanced_dataset2.json`: alternative training configuration.(IMDLBenCo Use the entire training set and test.)
   * `test_datasets.json`: evaluation sets for the main results.

3. Make sure the JSON format is unchanged; only modify the file paths.

For more details on dataset formats, please refer to the IMDL-BenCo documentation.

---

## 4. Training FRD-Net

The simplest way to start training is via the provided script:

```bash
bash runs/demo_train_FRDNet.sh
```



## 5. Evaluation on Standard Benchmarks

After training, place your best checkpoint in `weights/` or another directory of your choice.

The provided evaluation script:

```bash
bash runs/demo_test_FRDNet.sh
```



The evaluation computes pixel-level and image-level metrics (e.g. Pixel F1, Pixel AUC, Image F1, IoU) consistent with the results in the paper.


## Acknowledgement

Our implementation is mainly built upon the following codebases. We gratefully thank the authors for their wonderful open-source contributions.

[IMDLBenCo/](https://github.com/scu-zjz/IMDLBenCo), [FEC](https://github.com/guikunchen/FEC).

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026amplifying,
  title   = {Amplifying Discrepancies: Exploiting Macro and Micro Inconsistencies for Image Manipulation Localization},
  author  = {Chen, Shenghao and Zhao, Yibo and Wang, Tianyi and Ma, Chunjie and Guan, Weili and Li, Ming and Gao, Zan},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {42},
  pages   = {35357--35365},
  year    = {2026},
  doi     = {10.1609/aaai.v40i42.40844}
}
```
