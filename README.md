# Fraud Detection ML Pipeline
End-to-end machine learning project to detect fraudulent transactions using Random Forest and XGBoost, with model comparison and evaluation.

## Overview
This project builds an end-to-end machine learning pipeline to detect fraudulent transactions. It compares multiple models (Random Forest and XGBoost) to improve detection accuracy and reduce false positives.
The pipeline includes data preprocessing, feature scaling, model training, evaluation, and comparison using standard classification metrics.

## Problem Statement
Fraud detection systems often generate false-positive alerts. This project focuses on building a model that improves fraud detection accuracy while reducing unnecessary alerts.

## Key Highlights
- Built an end-to-end ML pipeline for fraud detection
- Compared multiple models (Random Forest vs XGBoost)
- Evaluated using Accuracy, Precision, Recall, and F1 Score
- Implemented feature scaling and train-test split
- Structured project following real-world ML workflow

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
| Model          | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|----------|--------|----------|
| Random Forest | 0.80     | 0.75     | 0.80   | 0.77     |
| XGBoost       | 0.90     | 0.85     | 0.90   | 0.87     |

The comparison shows that XGBoost outperforms Random Forest on the sample dataset, achieving higher precision and recall, making it more effective for fraud detection scenarios.

*Note: Results are based on sample dataset and may vary with real-world data.*

## How to Run
1. Install dependencies:
  pip install -r requirements.txt
2. Run the model:
  python src/train_model.py
