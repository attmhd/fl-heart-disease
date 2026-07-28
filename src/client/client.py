# ruff: noqa: E402
"""
Flower Federated Learning Client node service launcher.

This module initializes local datasets, creates the numpy clients, connects with
the Flower Server gRPC fleet, and logs distributed model updates to MLflow.
"""

import os
import sys
import time
import warnings

import flwr as fl
import mlflow
from flwr.compat.client.app import start_client_internal
from sklearn.exceptions import ConvergenceWarning

from configs import config
from data import dataset
from utils.client_utils import create_client

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def client_fn(context: fl.common.Context) -> fl.client.Client:
    """ClientApp factory function — called by flower-superexec / flower-supernode.

    Args:
        context (Context): Flower shared run context.
    """
    X_train, X_test, y_train, y_test = dataset.load_data(config.CLIENT_DATASET_PATH)
    client = create_client(X_train, y_train, X_test, y_test, config.CLIENT_ID)
    return client.to_client()


app = fl.client.ClientApp(client_fn=client_fn)


def start_client():
    """Legacy entry point — starts client directly via Python (for backward compatibility).

    Wraps the MLflow child run and starts the gRPC infrastructure inline.
    Kept for local/Docker Compose runs where flower-supernode is not separately deployed.
    """
    X_train, X_test, y_train, y_test = dataset.load_data(config.CLIENT_DATASET_PATH)
    client = create_client(X_train, y_train, X_test, y_test, config.CLIENT_ID)

    try:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                print(f"[{config.CLIENT_ID}] Connecting to Flower Server at {config.FL_CLIENT_SERVER_ADDRESS} (Attempt {attempt + 1}/{max_retries})...")
                start_client_internal(
                    server_address=config.FL_CLIENT_SERVER_ADDRESS,
                    node_config={},
                    client=client.to_client(),
                    insecure=True,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{config.CLIENT_ID}] Connection failed: {e}. Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    print(f"[{config.CLIENT_ID}] Could not connect to Flower Server. Max retries reached.")
                    raise e
    finally:
        if mlflow.active_run() is not None:
            print(f"[{config.CLIENT_ID}] Ending active MLflow child run...")
            mlflow.end_run()


if __name__ == "__main__":
    start_client()
