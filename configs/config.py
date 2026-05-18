import os

# MLflow Configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "FL_HeartDisease_LogReg_v1"

# Flower Server Configuration
FL_SERVER_ADDRESS = "0.0.0.0:8080"
FL_NUM_ROUNDS = 10

# Flower Client Configuration
FL_CLIENT_SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "fl_server:8080")
CLIENT_DATASET_PATH = os.environ.get("DATASET_PATH")
CLIENT_ID = os.environ.get("CLIENT_ID", "client_unknown")

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    return {"server_round": server_round}
