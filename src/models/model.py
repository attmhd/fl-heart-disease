"""
Model estimator generator and configuration manager.

This module initializes machine learning estimators (Logistic Regression)
with compatibility checks for solver types and regularization penalties.
"""

import os

from sklearn.linear_model import LogisticRegression


def _get_env_int(key, default):
    """Retrieve integer value from environment variables with safe fallback.

    Args:
        key (str): Target environment key.
        default (int): Fallback value.
    """
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key, default):
    """Retrieve float value from environment variables with safe fallback.

    Args:
        key (str): Target environment key.
        default (float): Fallback value.
    """
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_lr_model():
    """Build and return configured Logistic Regression model with solver compatibility rules.

    Saga/Liblinear support l1/l2; newton-cg/lbfgs/sag only support l2/None.
    """
    max_iter = _get_env_int("MODEL_MAX_ITER", 1)
    solver = os.environ.get("MODEL_SOLVER", "saga")
    c = _get_env_float("MODEL_C", 1.0)
    penalty = os.environ.get("MODEL_PENALTY", "l2")

    if penalty == "none":
        # Scikit-learn v1.2+ uses None instead of 'none'
        penalty = None

    return LogisticRegression(max_iter=max_iter, solver=solver, C=c, penalty=penalty, warm_start=True, random_state=42)
