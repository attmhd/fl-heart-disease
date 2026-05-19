from sklearn.linear_model import LogisticRegression


def get_lr_model():
    return LogisticRegression(
        penalty="l2",
        max_iter=1000,
        solver="lbfgs",
        warm_start=True,
        random_state=42
    )
