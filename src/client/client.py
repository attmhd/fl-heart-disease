import mlflow
import flwr as fl
import numpy as np
from src.data import dataset
from src.models import model
from src.utils import model_utils
from src.utils import data_utils
from configs import config
from datetime import datetime
from sklearn.metrics import accuracy_score, log_loss

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
        current_round = config.get("server_round", "unknown")

        # MLflow logging (nested per client)
        with mlflow.start_run(run_name=f"round_{current_round}", nested=True):
            mlflow.set_tag("mlflow.user", self.client_id)

            stats = data_utils.get_client_stats(self.y_train)

            mlflow.log_params({
                "client_id": self.client_id,
                "num_samples": stats["num_samples"],
                "class_0": stats["class_0"],
                "class_1": stats["class_1"]
            })

            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "param_norm": param_norm,
                "delta_norm": delta_norm
            })

        return local_weights, len(self.X_train), {
            "train_accuracy": train_acc,
            "local_loss": train_loss
        }

    def evaluate(self, parameters, config):
        model_utils.set_model_parameters(self.learner, parameters)

        probs = self.learner.predict_proba(self.X_test)
        preds = self.learner.predict(self.X_test)

        loss = log_loss(self.y_test, probs, labels=[0, 1])
        acc = accuracy_score(self.y_test, preds)

        return loss, len(self.X_test), {"accuracy": acc}


def start_client():
    X_train, X_test, y_train, y_test = dataset.load_data(config.CLIENT_DATASET_PATH)

    client = HeartDiseaseClient(
        X_train, y_train, X_test, y_test, config.CLIENT_ID
    )

    run_name = f"local_{config.CLIENT_ID}_{datetime.now().strftime('%Y-%m-%d')}"

    # Mulai Parent Run untuk sesi client ini
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.user", config.CLIENT_ID)
        mlflow.set_tag("handled_by", f"User {config.CLIENT_ID}")
        mlflow.set_tag("project", "Federated Heart Disease")
        mlflow.set_tag("mlflow.note.content", f"Partisipasi client {config.CLIENT_ID} dalam pelatihan Federated Learning untuk deteksi penyakit jantung.")
        
        fl.client.start_client(
            server_address=config.FL_CLIENT_SERVER_ADDRESS,
            client=client.to_client()
        )


if __name__ == "__main__":
    start_client()