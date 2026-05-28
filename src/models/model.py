import os

from sklearn.linear_model import LogisticRegression


def _get_env_int(key, default):
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key, default):
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_lr_model():
    max_iter = _get_env_int("MODEL_MAX_ITER", 1000)
    solver = os.environ.get("MODEL_SOLVER", "saga")
    c = _get_env_float("MODEL_C", 1.0)
    penalty = os.environ.get("MODEL_PENALTY", "l2")

    # Handle penalty compatibility with solver
    # if penalty is 'none' / None, we must handle it cleanly (none is supported by lbfgs, newton-cg, sag, saga)
    # saga/liblinear support l1/l2; newton-cg/lbfgs/sag only support l2/none (elasticnet is only saga)
    if penalty == "none":
        # Scikit-learn v1.2+ uses None instead of 'none'
        penalty = None

    return LogisticRegression(max_iter=max_iter, solver=solver, C=c, penalty=penalty, warm_start=False, random_state=42)
