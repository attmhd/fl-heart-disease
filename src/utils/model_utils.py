import numpy as np


def get_model_parameters(model):
    if getattr(model, "fit_intercept", True):
        return [model.coef_, model.intercept_]
    return [model.coef_]


def set_model_parameters(model, parameters):
    model.coef_ = parameters[0]
    if getattr(model, "fit_intercept", True):
        model.intercept_ = parameters[1]
    return model


def flatten_parameters(parameters):
    return np.concatenate([p.flatten() for p in parameters])


def set_initial_params(model, X_train, y_train):
    unique_classes = np.unique(y_train)

    indices = []
    for cls in unique_classes:
        cls_idx = np.where(y_train == cls)[0]
        indices.extend(cls_idx[:5])

    indices = np.array(indices)

    model.fit(X_train[indices], y_train[indices])
    return model
