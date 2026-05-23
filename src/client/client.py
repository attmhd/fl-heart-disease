import os
import sys
from datetime import datetime

# Ensure src directory is at the front of sys.path to avoid collision with root-level 'data' volume mount
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flwr as fl
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from configs import config
from data import dataset
from models import model
from utils import data_utils, eval_utils, model_utils

# MLflow setup
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test, client_id):
        self.client_id = client_id

        self.learner = model.get_lr_model()
        model_utils.set_initial_params(self.learner, X_train, y_train)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return model_utils.get_model_parameters(self.learner)

    def fit(self, parameters, config):
        model_utils.set_model_parameters(self.learner, parameters)

        # Global weights sebelum training
        global_weights = model_utils.get_model_parameters(self.learner)

        # Training lokal
        self.learner.fit(self.X_train, self.y_train)

        # Local weights setelah training
        local_weights = model_utils.get_model_parameters(self.learner)

        # Metrics
        probs = self.learner.predict_proba(self.X_train)
        preds = self.learner.predict(self.X_train)

        train_acc = accuracy_score(self.y_train, preds)
        train_loss = log_loss(self.y_train, probs, labels=[0, 1])

        # Parameter analysis
        global_flat = model_utils.flatten_parameters(global_weights)
        local_flat = model_utils.flatten_parameters(local_weights)

        param_norm = np.linalg.norm(local_flat)
        delta_norm = np.linalg.norm(local_flat - global_flat)

        # Ambil informasi ronde dari config (dikirim oleh server)
        current_round = config.get("server_round")
        if current_round is not None:
            try:
                current_round = int(current_round)
            except ValueError:
                current_round = 0
        else:
            current_round = 0

        # Dinamis start child run menggunakan server_run_id dari server jika belum aktif
        parent_run_id = config.get("server_run_id")
        if mlflow.active_run() is None and parent_run_id:
            run_name = f"Federated_Client_{self.client_id}_Logistic_Regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tags = {"mlflow.parentRunId": parent_run_id}
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self.mlflow_run_id = run.info.run_id

            # Set tags for user/handled_by/project
            mlflow.set_tag("mlflow.user", self.client_id)
            mlflow.set_tag("handled_by", f"User {self.client_id}")
            mlflow.set_tag("project", "Federated Heart Disease")
            mlflow.set_tag("mlflow.note.content", f"Partisipasi client {self.client_id} dalam pelatihan Federated Learning untuk deteksi penyakit jantung.")
        elif mlflow.active_run() is not None:
            self.mlflow_run_id = mlflow.active_run().info.run_id

        # Log parameters sekali saja di ronde pertama
        if current_round <= 1:
            stats = data_utils.get_client_stats(self.y_train)
            model_params = self.learner.get_params()
            log_dict = {
                "client_id": self.client_id,
                "num_samples": stats["num_samples"],
                "class_0": stats["class_0"],
                "class_1": stats["class_1"],
            }
            # Log parameter model dengan prefix model_
            for k, v in model_params.items():
                log_dict[f"model_{k}"] = str(v)
            mlflow.log_params(log_dict)

        # Log metrics langsung ke child run dengan step=current_round
        mlflow.log_metric("train_accuracy", train_acc, step=current_round)
        mlflow.log_metric("train_loss", train_loss, step=current_round)
        mlflow.log_metric("param_norm", param_norm, step=current_round)
        mlflow.log_metric("delta_norm", delta_norm, step=current_round)

        return (
            local_weights,
            len(self.X_train),
            {
                "train_accuracy": train_acc,
                "local_loss": train_loss,
                "param_norm": float(param_norm),
                "delta_norm": float(delta_norm),
            },
        )

    def evaluate(self, parameters, config):
        model_utils.set_model_parameters(self.learner, parameters)

        probs = self.learner.predict_proba(self.X_test)
        preds = self.learner.predict(self.X_test)

        loss = log_loss(self.y_test, probs, labels=[0, 1])
        acc = accuracy_score(self.y_test, preds)

        # Check if it is the final round for client-side distributed evaluation
        is_final_round = config.get("is_final_round", False)
        if isinstance(is_final_round, str):
            is_final_round = is_final_round.lower() == "true"

        if is_final_round:
            print(f"[{self.client_id}] Starting final evaluation on local dataset...")
            # Resubscribe/activate child run on this specific thread if it is not already active
            client_run = None
            if mlflow.active_run() is None:
                saved_run_id = getattr(self, "mlflow_run_id", None)
                if saved_run_id:
                    client_run = mlflow.start_run(run_id=saved_run_id)
                else:
                    # Fallback to create client run if it was never created in fit()
                    parent_run_id = config.get("server_run_id")
                    if parent_run_id:
                        run_name = f"Federated_Client_{self.client_id}_Logistic_Regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        tags = {"mlflow.parentRunId": parent_run_id}
                        client_run = mlflow.start_run(run_name=run_name, tags=tags)

                        mlflow.set_tag("mlflow.user", self.client_id)
                        mlflow.set_tag("handled_by", f"User {self.client_id}")
                        mlflow.set_tag("project", "Federated Heart Disease")
                        mlflow.set_tag(
                            "mlflow.note.content",
                            f"Partisipasi client {self.client_id} dalam pelatihan Federated Learning untuk deteksi penyakit jantung.",
                        )

            try:
                # Evaluate and log detailed metrics and plots using centralized eval_utils helper
                eval_utils.evaluate_and_log_metrics(
                    model_obj=self.learner,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    artifact_path="best_model_client_evaluation",
                    metrics_prefix="best_model_client",
                    title_suffix=f"- {self.client_id} (Best Model)",
                )
            finally:
                if client_run is not None:
                    mlflow.end_run()

            print(f"[{self.client_id}] Best Model Final Evaluation Completed: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

        return loss, len(self.X_test), {"accuracy": acc}


def client_fn(context: fl.common.Context) -> fl.client.Client:
    """ClientApp factory function — called by flower-superexec / flower-supernode."""
    X_train, X_test, y_train, y_test = dataset.load_data(config.CLIENT_DATASET_PATH)
    return HeartDiseaseClient(X_train, y_train, X_test, y_test, config.CLIENT_ID).to_client()


# Module-level ClientApp — loaded by flower-superexec (plugin-type=clientapp)
app = fl.client.ClientApp(client_fn=client_fn)


def start_client():
    """Legacy entry point — starts client directly via Python (for backward compatibility).

    Kept for local/Docker Compose runs where flower-supernode is not separately deployed.
    """
    X_train, X_test, y_train, y_test = dataset.load_data(config.CLIENT_DATASET_PATH)
    client = HeartDiseaseClient(X_train, y_train, X_test, y_test, config.CLIENT_ID)

    import time

    # Use the internal (non-deprecated) gRPC bidi client start function
    from flwr.compat.client.app import start_client_internal

    try:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                print(f"[{config.CLIENT_ID}] Connecting to Flower Server at {config.FL_CLIENT_SERVER_ADDRESS} (Attempt {attempt+1}/{max_retries})...")
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
