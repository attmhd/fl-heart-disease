from datetime import datetime

import flwr as fl
import mlflow

from configs import config
from models import model
from utils import model_utils

# MLflow setup
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
# mlflow.enable_system_metrics_logging() # Di-disable karena memerlukan paket psutil yang tidak terinstall


class MLflowStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_accuracy = -1.0
        self.best_weights = None
        self.best_round = None
        self.current_weights = None
        self.participant_clients = set()

    def _register_best_model(self, model_name):
        """Register, tag, and promote the best model to Production stage in MLflow Model Registry."""
        try:
            from mlflow import MlflowClient

            client = MlflowClient()

            # 1. Perbarui deskripsi model terdaftar
            client.update_registered_model(name=model_name, description="Model global hasil pembelajaran terfederasi menggunakan Flower untuk klasifikasi penyakit jantung (UCI Heart Disease Dataset).")

            # Dapatkan nomor versi model terbaru yang baru saja didaftarkan
            versions = client.get_latest_versions(model_name, stages=["None"])
            if versions:
                latest_version = versions[0].version

                # 2. Perbarui deskripsi untuk versi model ini
                client.update_model_version(
                    name=model_name,
                    version=latest_version,
                    description=f"Model terbaik hasil agregasi federated learning yang dipilih berdasarkan akurasi validasi teragregasi tertinggi pada Ronde {self.best_round} dengan Akurasi: {self.best_accuracy:.4f}.",
                )

                # 3. Tambahkan metadata tags untuk versi model
                client.set_model_version_tag(model_name, latest_version, "validation_accuracy", f"{self.best_accuracy:.4f}")
                client.set_model_version_tag(model_name, latest_version, "best_round", str(self.best_round))
                client.set_model_version_tag(model_name, latest_version, "algorithm", "Logistic Regression")
                client.set_model_version_tag(model_name, latest_version, "environment", "Production")
                client.set_model_version_tag(model_name, latest_version, "total_participating_clients", str(len(self.participant_clients)))

                # 4. Transisikan model ke stage 'Production' dan arsipkan versi lama
                print(f"Transitioning model version {latest_version} to Stage 'Production'...")
                client.transition_model_version_stage(name=model_name, version=latest_version, stage="Production", archive_existing_versions=True)

                # 5. Pasang aliases 'champion' dan 'Production' ke versi terbaik ini
                client.set_registered_model_alias(model_name, "champion", latest_version)
                client.set_registered_model_alias(model_name, "Production", latest_version)
                print(f"Successfully registered, tagged, and promoted Model Version {latest_version} to Stage 'Production' with alias '@Production'.")
        except Exception as e:
            print(f"Error during advanced model registry operations: {e}")

    def _log_client_fit_metrics(self, rnd, results):
        """Extract and log individual client fit metrics to MLflow before aggregation."""
        accs = []
        losses = []
        num_examples_list = []

        for client, res in results:
            cid = client.cid
            client_acc = res.metrics.get("train_accuracy")
            client_loss = res.metrics.get("local_loss")
            client_n = res.num_examples

            # Log metrik individual per client sebelum agregasi (pembuktian FedAvg)
            if client_acc is not None:
                mlflow.log_metric(f"client_{cid}_train_accuracy", client_acc, step=rnd)
                accs.append(client_acc)
            if client_loss is not None:
                mlflow.log_metric(f"client_{cid}_train_loss", client_loss, step=rnd)
                losses.append(client_loss)

            mlflow.log_metric(f"client_{cid}_fit_examples", client_n, step=rnd)
            num_examples_list.append(client_n)

        return accs, losses, num_examples_list

    def _log_client_eval_metrics(self, rnd, results):
        """Extract and log individual client evaluation metrics to MLflow before aggregation."""
        accs = []
        num_examples_list = []

        for client, res in results:
            cid = client.cid
            client_acc = res.metrics.get("accuracy")
            client_loss = res.loss
            client_n = res.num_examples

            # Log metrik individual per client sebelum agregasi (pembuktian FedAvg)
            if client_acc is not None:
                mlflow.log_metric(f"client_{cid}_val_accuracy", client_acc, step=rnd)
                accs.append(client_acc)
            if client_loss is not None:
                mlflow.log_metric(f"client_{cid}_val_loss", client_loss, step=rnd)

            mlflow.log_metric(f"client_{cid}_val_examples", client_n, step=rnd)
            num_examples_list.append(client_n)

        return accs, num_examples_list

    def aggregate_fit(self, rnd, results, failures):
        # 1. Agregasi bobot dari semua client
        aggregated_result = super().aggregate_fit(rnd, results, failures)

        if aggregated_result is not None:
            aggregated_weights, _ = aggregated_result
            # Simpan bobot ronde ini untuk evaluasi
            self.current_weights = aggregated_weights

            # Deteksi dan catat partisipan aktif pada ronde ini
            for client, _ in results:
                self.participant_clients.add(client.cid)

            # Log jumlah client aktif pada ronde ini ke MLflow
            mlflow.log_metric("active_fit_clients", len(results), step=rnd)

            accs, losses, num_examples_list = self._log_client_fit_metrics(rnd, results)

            # Log metrics ke parent run dengan menyertakan step=rnd
            if accs:
                # 1. Rata-rata Sederhana (Simple Average)
                simple_avg_acc = sum(accs) / len(accs)
                mlflow.log_metric("avg_train_accuracy", simple_avg_acc, step=rnd)

                # 2. Rata-rata Tertimbang (Weighted Average / FedAvg)
                total_examples = sum(num_examples_list)
                if total_examples > 0:
                    weighted_avg_acc = sum(a * n for a, n in zip(accs, num_examples_list)) / total_examples
                else:
                    weighted_avg_acc = simple_avg_acc
                mlflow.log_metric("fedavg_train_accuracy", weighted_avg_acc, step=rnd)

            if losses:
                # 1. Rata-rata Sederhana (Simple Average)
                simple_avg_loss = sum(losses) / len(losses)
                mlflow.log_metric("avg_train_loss", simple_avg_loss, step=rnd)

                # 2. Rata-rata Tertimbang (Weighted Average / FedAvg)
                total_examples = sum(num_examples_list)
                if total_examples > 0:
                    weighted_avg_loss = sum(l * n for l, n in zip(losses, num_examples_list)) / total_examples
                else:
                    weighted_avg_loss = simple_avg_loss
                mlflow.log_metric("fedavg_train_loss", weighted_avg_loss, step=rnd)

        return aggregated_result

    def aggregate_evaluate(self, rnd, results, failures):
        # 1. Agregasi hasil evaluasi dari semua client menggunakan super class
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)

        if aggregated_result is not None:
            loss_aggregated, metrics_aggregated = aggregated_result

            # Deteksi dan catat partisipan aktif pada ronde ini
            for client, _ in results:
                self.participant_clients.add(client.cid)

            # Log jumlah client aktif pada ronde ini ke MLflow
            mlflow.log_metric("active_evaluate_clients", len(results), step=rnd)

            accs, num_examples_list = self._log_client_eval_metrics(rnd, results)

            # Log metrics global model ke parent run dengan menyertakan step=rnd
            if loss_aggregated is not None:
                mlflow.log_metric("global_val_loss", loss_aggregated, step=rnd)

            if accs:
                # Menggunakan weighted average untuk accuracy berdasarkan jumlah sampel/data uji di tiap client
                total_examples = sum(num_examples_list)
                if total_examples > 0:
                    weighted_acc = sum(a * n for a, n in zip(accs, num_examples_list)) / total_examples
                else:
                    weighted_acc = sum(accs) / len(accs)

                # Log global_val_accuracy dan fedavg_val_accuracy secara konsisten
                mlflow.log_metric("global_val_accuracy", weighted_acc, step=rnd)
                mlflow.log_metric("fedavg_val_accuracy", weighted_acc, step=rnd)

                # Cek dan simpan model terbaik berdasarkan akurasi tertinggi
                if weighted_acc > self.best_accuracy:
                    self.best_accuracy = weighted_acc
                    self.best_weights = self.current_weights
                    self.best_round = rnd
                    print(f"[ROUND {rnd}] New best global model achieved! Accuracy: {weighted_acc:.4f}")

            # Hanya log model terbaik & daftarkan ke Model Registry di ronde terakhir
            if rnd == config.FL_NUM_ROUNDS:
                weights_to_log = self.best_weights if self.best_weights is not None else self.current_weights

                if weights_to_log is not None:
                    # Reconstruct best global model
                    ndarrays = fl.common.parameters_to_ndarrays(weights_to_log)
                    lr_model = model.get_lr_model()
                    model_utils.set_model_parameters(lr_model, ndarrays)

                    # Log model terbaik ke MLflow dan daftarkan ke Model Registry
                    model_name = config.MLFLOW_MODEL_NAME
                    print(f"Logging the best global model from Round {self.best_round} (Accuracy: {self.best_accuracy:.4f}) to MLflow Model Registry...")
                    mlflow.sklearn.log_model(sk_model=lr_model, artifact_path="model", registered_model_name=model_name)

                    # Daftarkan parameter model terbaik ke MLflow parent run
                    mlflow.log_params({"best_model_round": self.best_round, "best_model_val_accuracy": self.best_accuracy, "total_participating_clients": len(self.participant_clients)})

                    # Pengelolaan Model Registry tingkat lanjut menggunakan MlflowClient
                    self._register_best_model(model_name)
                else:
                    print("Warning: No weights found to reconstruct global model for MLflow logging.")

        return aggregated_result


def start_server():
    run_name = f"Federated_Server_Logistic_Regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.user", "server")
        mlflow.set_tag("handled_by", "Admin Server")
        mlflow.set_tag("project", "Federated Heart Disease")
        mlflow.set_tag("mlflow.note.content", "Eksperimen FL Murni: Server tidak memiliki akses ke dataset. Evaluasi hanya berdasarkan laporan metrik dari client.")

        # Log server configurations
        mlflow.log_params({"num_rounds": config.FL_NUM_ROUNDS, "strategy": "FedAvg"})

        strategy = MLflowStrategy(on_fit_config_fn=config.fit_config)

        fl.server.start_server(server_address=config.FL_SERVER_ADDRESS, config=fl.server.ServerConfig(num_rounds=config.FL_NUM_ROUNDS), strategy=strategy)


if __name__ == "__main__":
    start_server()
