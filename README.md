# Hyperparameter Tuning Project

## Overview
This project demonstrates hyperparameter optimization for an XGBoost classifier using Optuna. It applies Bayesian Optimization with 5-fold cross-validation to find the best model parameters for a binary classification problem.

## Features
- Data Loading and Preprocessing
- XGBoost Classifier
- Optuna Hyperparameter Tuning
- 5-Fold Cross Validation
- Model Evaluation
- Save Best Model and Parameters
- Optimization History Visualization

## Project Structure

```
hyperparameter_tuning_project/
│── data/
│── models/
│── outputs/
│── plots/
│── src/
│── main.py
│── requirements.txt
│── README.md
```

## Installation

```bash
git clone <repository-url>
cd hyperparameter_tuning_project
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- Matplotlib
- Joblib

## Output

- Best Hyperparameters
- Trained Model (`best_model.pkl`)
- Optuna Study (`study.pkl`)
- Optimization History Plot
- Model Evaluation Metrics

