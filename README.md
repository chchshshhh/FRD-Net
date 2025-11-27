# FRD-Net for Image Manipulation Localization  
**AAAI 2026 – Amplifying Discrepancies: Exploiting Macro and Micro Inconsistencies for Image Manipulation Localization **

This repository contains the code and configuration files used in our AAAI 2026 paper:

> **“Amplifying Discrepancies: Exploiting Macro and Micro Inconsistencies for Image Manipulation Localization”**

Our method, **FRD-Net**, is implemented on top of the **IMDL-BenCo** codebase, a comprehensive benchmark and framework for image manipulation detection and localization.

This package is intended as **supplementary material** to help reviewers and researchers reproduce the main results and ablation studies in the paper.

---

## 1. Repository Structure

At the top level:

- `IMDLBenCo/`  
  Core benchmark code (datasets, transforms, evaluation metrics, training utilities, model registry, etc.)

- `IMDLBenCo/modules/backbones/FRDNet.py`  
  Implementation of **FRD-Net**, our backbone that exploits macro and micro inconsistencies.

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

- `setup.py`, `MANIFEST.in`  
  Packaging scripts for installing IMDL-BenCo/FRD-Net as a local Python package.

- `LICENSE`  
  License file (Creative Commons Attribution 4.0 International).

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

# 3) Install IMDL-BenCo + FRD-Net as an editable package
pip install -e .
