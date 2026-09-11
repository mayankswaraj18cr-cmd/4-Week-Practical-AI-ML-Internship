"""Reusable sklearn estimators for supervised learning."""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Build a numeric imputation, scaling, and Ridge regression pipeline."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])


def build_random_forest_pipeline(n_estimators: int = 100, random_state: int = 42) -> Pipeline:
    """Build a mixed-type preprocessing and random forest classifier."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric, [0]), ("categorical", categorical, [1])])
    return Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))])


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate MSE and coefficient of determination."""
    y_true, y_pred = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("actual and predicted must have equal, non-empty shape")
    mse = float(np.mean((y_true - y_pred) ** 2))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mse": mse, "r2": 1.0 if denominator == 0 and mse == 0 else 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator}
