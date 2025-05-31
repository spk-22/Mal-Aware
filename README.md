# Mal-Aware

## Overview

This project implements a machine learning model for classifying software as either benign or malicious based on its system process behavior. By analyzing various features derived from running processes, the model aims to provide an effective method for identifying malware.

## Project Goals

* Develop a machine learning model capable of accurately classifying malware.
* Explore and analyze key system process features relevant to malware detection.
* Provide a clear and reproducible workflow for training and evaluating the model.
* (Optional: Investigate the interpretability of the model's predictions.)

## Dataset

The CICIDS-2017 dataset used in this project contains information about system processes, with features such as process counts, memory usage, loaded libraries (DLLs), handles, and more. The dataset is labeled to indicate whether a process or system state is benign or associated with malware.


## Methodology

The project typically involves the following steps:

1.  **Data Loading and Exploration:** Loading the system process dataset and understanding its structure and features.
2.  **Feature Preprocessing:** Cleaning, scaling, and potentially transforming the features to prepare them for the machine learning model.
3.  **Model Selection:** Choosing an appropriate machine learning model for classification (e.g., Neural Network, Random Forest, Gradient Boosting).
4.  **Model Training:** Training the selected model on the preprocessed data.
5.  **Model Evaluation:** Assessing the performance of the trained model using appropriate metrics (e.g., accuracy, precision, recall, F1-score, confusion matrix, AUC-ROC).


## Code Structure

*(Provide a brief overview of the files in your repository, e.g.,)*

* `malware.py` : Contains the main code for data loading, preprocessing, model training, and evaluation.
* `malware_model.pth` : The trained model with model weights in .pth format.
* `sampling_comparison_simluation.html`: To visulaize the difference between casual sampling and random sampling.
* `README.md`: This file.
* `models.py`,`utils.py`,`web-app.py`: Visual Interface to load files and check model predictions, evaluation, accuracy metrics and reasoning.

## Requirements - Python libraries:

* Python 3.x
* pandas
* numpy
* scikit-learn
* (Potentially other libraries e.g., TensorFlow, PyTorch, XGBoost)

**(If you have a `requirements.txt` file, mention it here and how to install dependencies.)**

```bash
pip install -r requirements.txt
Usage
Clone the repository:
Bash

git clone <repository_url>
Navigate to the project directory:
Bash

cd your_repository_name
(If applicable) Install dependencies:
Bash

pip install -r requirements.txt
Run the main script or notebook:
Bash

jupyter notebook malware_classification.ipynb
# or
python malware_classification.py
Results
(Once you have results, you can add a section here to summarize the performance of your model on the malware classification task. Include key metrics like accuracy, F1-score, etc.)

Future Work
Experiment with different machine learning models and hyperparameter tuning.
Explore advanced feature engineering techniques.
Investigate the interpretability of the model.
Evaluate the model on more diverse and real-world malware samples.
(Potentially integrate with a system monitoring tool.)
