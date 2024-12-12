# EEG Seizure Classification for TSC Model

This project is designed to classify seizures using advanced data preprocessing, machine learning models, and deep learning techniques. Its goal is to provide accurate and interpretable predictions to support EEG annotations.

![Python](https://img.shields.io/badge/python-v3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-blue)
## Overview
This project aims to develop a machine learning classifier to detect seizures in EEG data based on 30-second segments of recorded signals. The data is derived from mouse Tuberous Sclerosis Complex (TSC) mouse models. This genetic disorder causes the growth of benign tumors in various organs and is a common cause of epilepsy in humans. TSC is characterized by the presence of non-cancerous growths, or hamartomas, that can develop in the brain, kidneys, heart, and other organs, often leading to neurological issues such as seizures. In humans, TSC-related epilepsy is difficult to manage, and the frequency of seizures can significantly affect the quality of life.

The data used in this project was collected from TSC knockout (KO) mice at the University of Vermont. These mouse models are vital for studying TSC-related epilepsy and offer insights into the progression of seizures and their underlying mechanisms. This project aims to expedite the detection of seizures in new EEG recordings, which will contribute to refining the characterization of seizure phenotypes in these mouse models. By doing so, we aim to support more accurate and timely diagnosis and improve the overall understanding of TSC-related epilepsy.

Given the large volume of EEG data collected, manual annotation is increasingly inefficient and impractical. To address this challenge, this project employs advanced machine learning and deep learning techniques to automate the detection of seizures, replacing a simple algorithm that currently identifies candidate seizure events based on arclength and spike counting. Through systematic exploration of various modeling and data preprocessing approaches, the project seeks to determine the most effective and computationally efficient model for seizure classification.

Ultimately, the objective is to create a robust, interpretable classifier that can handle large volumes of data and provide accurate predictions, enabling better management and understanding of TSC-related epilepsy. The data for these notebooks is not publicly available.
  
## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)    
3. [Methods](#methods)  
    - [Data Preprocessing](#data-preprocessing)  
    - [Model Training](#model-training)  
    - [Saliency Analysis](#saliency-analysis)  
4. [Results](#results)
---

## Dependencies

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



## Methods 
### Data Preprocessing
The data loaded into the notebooks in this repo was segmented and labeled based on manual annotations. In addition the EEG signals were normalized using the Median Absolute Deviation (MAD) and filtered using a bandpass Butterworth filter with cutoffs of 0.5 and 50 Hz and a filter order of four.

The first two CNNs trained used the filtered normalized data directly. The LSTM, Transformer, and final CNN implemented were trained on additional processed data. First, the data was randomly downsampled. This reduced the number of nonseizure windows, which initially was over 50,000, compared to the 417 seizure windows. Next, PCA was performed on the reduced data, which had around 4,500 samples in total. After the PCA transform, the Synthetic Minority Oversampling Technique (SMOTE) was used to increase the seizure class size. The LSTM, Transformer adn last CNN of the notebook are all trained using this augmented dataset. 

### Model Training
#### EEGCNN1D Model Architecture
The **EEGCNN1D** model is a 1D convolutional neural network designed for processing raw EEG data to classify or predict seizure events.
- **Input Layer**: 
  - 1D EEG signal with a single channel.
- **Convolutional Blocks**:
  - **Conv1**: 
    - 16 output channels, kernel size 5, stride 1, padding 2
    - Followed by ReLU and max pooling
  - **Conv2**: 
    - 32 output channels, similar kernel size, stride, and padding
    - Followed by ReLU and max pooling
  - **Conv3**: 
    - 64 output channels, similar structure
  - **Conv4**: 
    - 64 output channels, followed by ReLU and max pooling
- **Fully Connected Layers**:
  - **FC1**: 
    - 128 units, ReLU activation
  - **FC2**: 
    - Single output unit for classification
- **Dropout**: 
  - 0.5 rate to prevent overfitting
- **Loss Function and Optimizer**:
  - **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
  - **Optimizer**: Adam optimizer with a learning rate of 0.001
- **Training**:
  - **Number of Epochs**: 10
  
#### EEGCNN1D with Weighted Loss
- **Model Initialization**:
  - `EEGCNN1D` model is instantiated with the same structure as above
- **Loss Function**:
  - **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
  - **Weights**: 
    - `weights = 1.0 for class 0, 20.0 for class 1`
- **Optimizer**:
  - **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Training**:
  - **Number of Epochs**: 10

#### EEGLSTM Model Architecture
The **EEGLSTM** model is a Long Short-Term Memory (LSTM) network designed for processing sequential EEG data to classify or predict seizure events.
- **Input Layer**: 
  - 1024 features per time step in the EEG sequence.
- **LSTM Block**:
  - **LSTM**: 
    - Hidden size: 64 units, 1 layer.
    - Dropout: 0.3 applied to LSTM output.
- **Fully Connected Output**:
  - **FC**: 
    - A fully connected layer with 1 output unit for classification.
- **Loss Function and Optimizer**:
  - **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) with balanced class weights.
  - **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Training**:
  - **Number of Epochs**: 20

#### EEGTRANSFORMER Model Architecture
The **EEGTRANSFORMER** model is a transformer-based architecture designed for processing sequential EEG data to classify or predict seizure events.
- **Input Layer**: 
  - 1024 features per time step in the EEG sequence.
- **Transformer Block**:
  - **Transformer Encoder**: 
    - Dimensionality: 1024
    - Number of heads: 4
    - Feedforward dimension: 4 * dim_model
    - Activation: ReLU
    - Dropout: 0.1
    - 2 encoder layers with LayerNorm
- **Fully Connected Output**:
  - **FC**: 
    - A fully connected layer with 1 output unit for classification.
- **Loss Function and Optimizer**:
  - **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) with class weights for seizure events.
  - **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Training**:
  - **Number of Epochs**: 20
  
#### EEGCNNPCA1D Model Architecture
The **EEGCNNPCA1D** model is a 1D convolutional neural network designed for processing raw EEG data to classify or predict seizure events, similar to the previous CNN but with one less convolutional layer.
- **Input Layer**: 
  - 1D EEG signal with a single channel.
- **Convolutional Blocks**:
  - **Conv1**: 
    - 16 output channels, kernel size 5, stride 1, padding 2
    - Followed by ReLU and max pooling
  - **Conv2**: 
    - 32 output channels, similar kernel size, stride, and padding
    - Followed by ReLU and max pooling
  - **Conv3**: 
    - 64 output channels, similar structure, followed by ReLU and max pooling
- **Fully Connected Layers**:
  - **FC1**: 
    - 128 units, ReLU activation
  - **FC2**: 
    - Single output unit for classification
- **Dropout**: 
  - 0.5 rate to prevent overfitting
- **Loss Function and Optimizer**:
  - **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) with class weights for seizure events.
  - **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Training**:
  - **Number of Epochs**: 10 

### Saliency Analysis
After all models had been trained, five samples of each class were loaded into a new notebook, and the best version of each model was initialized. Saliency maps were generated for sample inputs and plotted along with the data input for comparison. The results are shown below. 

### Results
Model training results for each model type tested. Cross-validation was not performed, but models were relatively stable across several training attempts. Loss curves, confusion matrices, and saliency plots are shown below. 

#### EEGCNN1D Model 
![CNN1_CM](Result%20Images/CNN%20Mod%20Data%20CM.png)


#### EEGCNN1D with Weighted Loss


#### EEGLSTM Model Architecture

#### EEGTRANSFORMER Model Architecture

  
#### EEGCNNPCA1D Model Architecture






  

