# Customer Churn Prediction with Deep Neural Networks

Deep neural network pipeline for predicting customer churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which contains 7,043 customer records across 20 features including demographics, account information, and service subscriptions.

## Overview

Two DNN architectures are built and compared:

- **Baseline DNN** — 3 hidden layers, Adam optimizer, 50 epochs
- **Tuned DNN** — 5 hidden layers with Batch Normalization and Dropout, SGD optimizer, 100 epochs

## Results

| Metric | Baseline Model | Tuned Model |
|---|---|---|
| Validation Accuracy | 75.7% | **80.7%** |
| Test Accuracy | 75.9% | **80.0%** |
| Test Recall | 58.0% | **65.8%** |
| Test F1-Score | 0.561 | **0.577** |

Adding Batch Normalization and Dropout resolved the overfitting observed in the baseline model and drove the core validation accuracy improvement from 75.7% to 80.7%.

## Dataset

`WA_Fn-UseC_-Telco-Customer-Churn.csv` — upload to the same directory as the notebook before running.

## Tech Stack

Python · pandas · scikit-learn · TensorFlow/Keras · matplotlib · seaborn
