# EEG Seizure Classification for TSC Model

This project is designed for the classification of seizures using advanced data preprocessing, machine learning models, and deep learning techniques. The goal is to provide accurate and interpretable predictions to support EEG annotations.

---
![Python](https://img.shields.io/badge/python-v3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-blue)

## 📋 Table of Contents
- [Dependencies](#dependencies)
- [Overview](#overview)
- [Methods](#methods)
- [results](#results)
---

## 🔧 Dependencies

The following libraries and tools are required to run this project:

- **Core Libraries**:
  - `numpy`
  - `os`

- **PyTorch**:
  - `torch`
  - `torch.nn`
  - `torch.optim`
  - `torch.utils.data`

- **Scikit-learn**:
  - `sklearn.model_selection` (for `train_test_split`)
  - `sklearn.metrics` (for `confusion_matrix`, `classification_report`)
  - `sklearn.decomposition` (for `PCA`)

- **Data Sampling**:
  - `imblearn.under_sampling.RandomUnderSampler`
  - `imblearn.over_sampling.SMOTE`

- **Visualization**:
  - `matplotlib.pyplot`
  - `seaborn`

- **Other Tools**:
  - `google.colab` (if using Google Colab)
  - `joblib`

---

![Python](https://img.shields.io/badge/python-v3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Sklearn](

---

## Dependencies

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Matplotlib
- TorchCAM
- Scikit-learn


# 

## Overview
This project aims to create a classifier to detect seizures for EEG data based on 30-second clips of recordings. The data used is EEG data collected from mouse models of tuberous sclerosis complex (TSC), which is a common cause of epilepsy in humans. The data was collected at the University of Vermont from TSC KO mice. The goal is to expedite the detection of seizures in new recordings in order to develop a better phenotype for this mouse model. Additional data is being collected in large volumes and requires a more efficient sorting method other than manual annotation. 
