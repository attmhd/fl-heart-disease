"""
Federated Learning global configuration manager.

This module houses MLflow metrics logging tracking parameters, centralized Flower
gRPC server setups, and client environment configurations parsed dynamically at runtime.
"""

import os

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "FL_Heart_Disease"
MLFLOW_MODEL_NAME = "Heart_Disease_Global_Model"
FL_SERVER_ADDRESS = "0.0.0.0:8080"


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


FL_NUM_ROUNDS = _get_env_int("FL_NUM_ROUNDS", 10)
FL_FRACTION_FIT = _get_env_float("FL_FRACTION_FIT", 1.0)
FL_FRACTION_EVALUATE = _get_env_float("FL_FRACTION_EVALUATE", 1.0)
FL_MIN_FIT_CLIENTS = _get_env_int("FL_MIN_FIT_CLIENTS", 2)
FL_MIN_EVALUATE_CLIENTS = _get_env_int("FL_MIN_EVALUATE_CLIENTS", 2)
SERVER_DATASET_PATH = os.environ.get("SERVER_DATASET_PATH", "data/processed/heart_disease_cleveland_cleaned.csv")


FL_CLIENT_SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "fl_server:8080")
CLIENT_DATASET_PATH = os.environ.get("DATASET_PATH")
CLIENT_ID = os.environ.get("CLIENT_ID", "client_unknown")


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    return {"server_round": server_round}
