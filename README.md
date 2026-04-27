# Fraud Detection ML Pipeline

## Overview
This project builds an end-to-end machine learning pipeline to detect fraudulent transactions using Python, Scikit-learn, XGBoost, and LightGBM. This project compares multiple machine learning models to improve fraud detection accuracy and reduce false positives.

## Problem Statement
Fraud detection systems often generate false-positive alerts. This project focuses on building a model that improves fraud detection accuracy while reducing unnecessary alerts.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn

## Project Workflow
1. Load transaction data
2. Clean and preprocess data
3. Perform feature engineering
4. Train fraud detection models
5. Evaluate model performance
6. Compare model results

## Project Structure
```text
fraud-detection-ml-pipeline/
│── data/
│── notebooks/
│── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│── requirements.txt
│── README.md      
```

## Dataset
This project uses a sample fraud detection dataset with features and fraud labels.

## Model Evaluation
The project evaluates fraud detection models using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
Two models are compared:
- Random Forest
- XGBoost

## Results
### Random Forest
Accuracy: 0.80  
Precision: 0.75  
Recall: 0.80  
F1 Score: 0.77  
### XGBoost
Accuracy: 0.90  
Precision: 0.85  
Recall: 0.90  
F1 Score: 0.87  

*Note: Results are based on sample dataset and may vary with real-world data.*
