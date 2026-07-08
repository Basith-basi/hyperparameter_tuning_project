import optuna
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def get_search_space(trial):
    """
    Define the hyperparameter search space.
    """

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        ),

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0,
            5
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0,
            5
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0,
            5
        ),

        "objective": "binary:logistic",

        "eval_metric": "logloss",

        "random_state": 42,

        "n_jobs": -1,

        "verbosity": 0
    }

    return params


def objective(trial, X_train, y_train):
    """
    Objective function for Optuna.
    """

    params = get_search_space(trial)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = []

    for train_idx, valid_idx in cv.split(X_train, y_train):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[valid_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[valid_idx]

        model = XGBClassifier(**params)

        model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)


        predictions = model.predict(X_val)

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        scores.append(accuracy)

    return np.mean(scores)