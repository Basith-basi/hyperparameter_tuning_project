from xgboost import XGBClassifier


def create_model():
    """
    Create and return an XGBoost classifier.
    """

    model = XGBClassifier(
        random_state=42,
        objective="binary:logistic",   # Use "multi:softprob" for multiclass
        eval_metric="logloss",
        n_jobs=-1
    )

    return model