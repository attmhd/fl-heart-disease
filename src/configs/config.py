import os

# MLflow Configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "FL_Heart_Disease"
MLFLOW_MODEL_NAME = "Heart_Disease_Global_Model"

# Flower Server Configuration
FL_SERVER_ADDRESS = "0.0.0.0:8080"


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


FL_NUM_ROUNDS = _get_env_int("FL_NUM_ROUNDS", 10)
FL_FRACTION_FIT = _get_env_float("FL_FRACTION_FIT", 1.0)
FL_FRACTION_EVALUATE = _get_env_float("FL_FRACTION_EVALUATE", 1.0)
FL_MIN_FIT_CLIENTS = _get_env_int("FL_MIN_FIT_CLIENTS", 2)
FL_MIN_EVALUATE_CLIENTS = _get_env_int("FL_MIN_EVALUATE_CLIENTS", 2)
SERVER_DATASET_PATH = os.environ.get("SERVER_DATASET_PATH", "data/processed/heart_disease_hungary_cleaned.csv")

# Flower Client Configuration
FL_CLIENT_SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "fl_server:8080")
CLIENT_DATASET_PATH = os.environ.get("DATASET_PATH")
CLIENT_ID = os.environ.get("CLIENT_ID", "client_unknown")


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    return {"server_round": server_round}
