import os
from datetime import datetime

import flwr as fl
import mlflow
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

from configs import config as global_config
from models import model
from utils import data_utils, eval_utils, model_utils


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

    def fit(self, parameters, config):  # noqa: PLR0915
        import time

        model_utils.set_model_parameters(self.learner, parameters)

        # Training lokal dengan pencatatan durasi
        start_time = time.time()
        self.learner.fit(self.X_train, self.y_train)
        fit_duration = (time.time() - start_time) * 1000  # in milliseconds

        # Local weights setelah training
        local_weights = model_utils.get_model_parameters(self.learner)

        # Training Metrics
        train_probs = self.learner.predict_proba(self.X_train)
        train_preds = self.learner.predict(self.X_train)

        train_acc = accuracy_score(self.y_train, train_preds)
        train_loss = log_loss(self.y_train, train_probs, labels=[0, 1])
        train_precision = precision_score(self.y_train, train_preds, zero_division=0)
        train_recall = recall_score(self.y_train, train_preds, zero_division=0)

        # Validation Metrics (pada data uji lokal)
        val_probs = self.learner.predict_proba(self.X_test)
        val_preds = self.learner.predict(self.X_test)

        val_acc = accuracy_score(self.y_test, val_preds)
        val_loss = log_loss(self.y_test, val_probs, labels=[0, 1])
        val_precision = precision_score(self.y_test, val_preds, zero_division=0)
        val_recall = recall_score(self.y_test, val_preds, zero_division=0)

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

        # Log training metrics per ronde
        mlflow.log_metric("train_accuracy", train_acc, step=current_round)
        mlflow.log_metric("train_loss", train_loss, step=current_round)
        mlflow.log_metric("train_precision", train_precision, step=current_round)
        mlflow.log_metric("train_recall", train_recall, step=current_round)

        # Log validation metrics per ronde
        mlflow.log_metric("val_accuracy", val_acc, step=current_round)
        mlflow.log_metric("val_loss", val_loss, step=current_round)
        mlflow.log_metric("val_precision", val_precision, step=current_round)
        mlflow.log_metric("val_recall", val_recall, step=current_round)

        mlflow.log_metric("fit_duration", fit_duration, step=current_round)

        return (
            local_weights,
            len(self.X_train),
            {
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "train_precision": float(train_precision),
                "train_recall": float(train_recall),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "fit_duration": float(fit_duration),
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
            best_round_info = config.get("best_round")
            best_round_str = f" from Round {best_round_info}" if best_round_info is not None else ""
            print(f"[{self.client_id}] Starting final evaluation on local dataset for best model{best_round_str}...")
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
                dataset_name = os.path.basename(global_config.CLIENT_DATASET_PATH) if global_config.CLIENT_DATASET_PATH else "Unknown Dataset"
                eval_utils.evaluate_and_log_metrics(
                    model_obj=self.learner,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    artifact_path="best_model_client_evaluation",
                    metrics_prefix="best_model_client",
                    step=best_round_info if best_round_info is not None else config.get("server_round"),
                    title_suffix=f"- {self.client_id} (Best Model{best_round_str})\nDataset: {dataset_name}",
                )
            finally:
                if client_run is not None:
                    mlflow.end_run()

            print(f"[{self.client_id}] Best Model{best_round_str} Final Evaluation Completed: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

        return loss, len(self.X_test), {"accuracy": acc}


def create_client(X_train, y_train, X_test, y_test, client_id):
    """Factory function to build and return a clean HeartDiseaseClient client."""
    return HeartDiseaseClient(X_train, y_train, X_test, y_test, client_id)
