# Seizure Classification: EEG Seizure Classification for TSC Model

This project is designed to classify seizures using advanced data preprocessing, machine learning models, and deep learning techniques. Its goal is to provide accurate and interpretable predictions to support EEG annotations.

![Python](https://img.shields.io/badge/python-v3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-blue)
## Overview
This project aims to develop a machine learning classifier to detect seizures in EEG data based on 30-second segments of recorded signals. The data is derived from mouse Tuberous Sclerosis Complex (TSC) mouse models. This genetic disorder causes the growth of benign tumors in various organs and is a common cause of epilepsy in humans. TSC is characterized by the presence of non-cancerous growths, or hamartomas, that can develop in the brain, kidneys, heart, and other organs, often leading to neurological issues such as seizures. In humans, TSC-related epilepsy is difficult to manage, and the frequency of seizures can significantly affect the quality of life.

The data used in this project was collected from TSC knockout (KO) mice at the University of Vermont. These mouse models are vital for studying TSC-related epilepsy and offer insights into the progression of seizures and their underlying mechanisms. This project aims to expedite the detection of seizures in new EEG recordings, which will contribute to refining the characterization of seizure phenotypes in these mouse models. By doing so, we aim to support more accurate and timely diagnosis and improve the overall understanding of TSC-related epilepsy.

Given the large volume of EEG data collected, manual annotation is increasingly inefficient and impractical. To address this challenge, this project employs advanced machine learning and deep learning techniques to automate the detection of seizures, replacing a simple algorithm that currently identifies candidate seizure events based on arclength and spike counting. Through systematic exploration of various modeling and data preprocessing approaches, the project seeks to determine the most effective and computationally efficient model for seizure classification.

Ultimately, the objective is to create a robust, interpretable classifier that can handle large volumes of data and provide accurate predictions, enabling better management and understanding of TSC-related epilepsy.
  
## Table of Contents
1. [Dependencies](#dependencies)    
2. [Methods](#methods)  
    - [Data Preprocessing](#data-preprocessing)  
    - [Model Training](#model-training)  
    - [Saliency Analysis](#saliency-analysis)  
3. [Results](#results)
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
#### EEGCNN1D
The EEGCNN1D model is a simple 1D convolutional neural network (CNN) designed for processing raw EEG data to classify or predict seizure events. The model is structured as follows:

Model Architecture:
Input Layer:

The model expects a 1D sequence of EEG data as input. The input data has a single channel (1D signal) and is processed through a series of convolutional layers to extract relevant features.
Convolutional Blocks:

The network consists of four convolutional layers:
Conv1: 1D convolutional layer with 16 output channels, kernel size of 5, stride of 1, and padding of 2. This is followed by a ReLU activation function and max pooling with a kernel size of 2 and stride of 2.
Conv2: Another 1D convolutional layer with 32 output channels and similar kernel size and padding. ReLU activation and max pooling are applied.
Conv3: This layer has 64 output channels and follows the same structure as the previous convolutional layers.
Conv4: Another 1D convolutional layer with 64 output channels, which is followed by ReLU and max pooling.
These convolutional layers progressively extract features from the EEG signal, reducing its dimensionality through pooling operations and increasing the number of feature maps as the network deepens.

Fully Connected (FC) Layers:

After the convolutional layers, the output is flattened and passed through two fully connected layers:
FC1: A fully connected layer with 128 units, followed by a ReLU activation function.
FC2: The final output layer with a single unit, which outputs a value that can be interpreted as the classification or prediction for the given input.
Dropout: A dropout layer with a rate of 0.5 is included to prevent overfitting by randomly dropping neurons during training.

Loss Function and Optimizer:
Loss Function: The model uses Binary Cross-Entropy with Logits (BCEWithLogitsLoss), which is suitable for binary classification tasks. This loss function combines a sigmoid layer and the binary cross-entropy loss into one class, simplifying the implementation of classification tasks with one output unit.

Optimizer: The model is trained using the Adam optimizer, which is an adaptive learning rate optimizer. It is widely used for training deep learning models due to its efficiency and ease of use. The learning rate is set to 0.001.

### Saliency Analysis







  

