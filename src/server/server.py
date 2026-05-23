import os
from datetime import datetime

import flwr as fl
import mlflow

from configs import config
from data import dataset
from models import model
from utils import eval_utils, model_utils

# MLflow setup
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


class MLflowStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_accuracy = -1.0
        self.best_weights = None
        self.best_round = None
        self.current_weights = None
        self.participant_clients = set()

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        # Jika ronde terakhir dan kita memiliki best_weights, distribusikan best_weights!
        if server_round == config.FL_NUM_ROUNDS and self.best_weights is not None:
            print(f"[SERVER] Distributing the BEST global model weights from Round {self.best_round} (Accuracy: {self.best_accuracy:.4f}) for final client-side evaluation.")
            parameters_to_distribute = self.best_weights
        else:
            parameters_to_distribute = parameters

        return super().configure_evaluate(server_round, parameters_to_distribute, client_manager)

    def evaluate(self, server_round: int, parameters: fl.common.Parameters):
        # 1. Jalankan evaluasi sentral menggunakan super class (memanggil evaluate_fn pada server)
        eval_res = super().evaluate(server_round, parameters)

        if eval_res is not None:
            loss, metrics = eval_res
            centralized_acc = metrics.get("accuracy")
            if centralized_acc is not None:
                # Cek dan simpan model terbaik berdasarkan akurasi sentral tertinggi
                if centralized_acc > self.best_accuracy:
                    self.best_accuracy = centralized_acc
                    self.best_weights = parameters
                    self.best_round = server_round
                    print(f"[SERVER - CENTRALIZED] New best global model achieved at Round {server_round}! Centralized Accuracy: {centralized_acc:.4f}")
        return eval_res

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

        for idx, (client, res) in enumerate(results):
            cid = f"client_{idx + 1}"
            client_acc = res.metrics.get("train_accuracy")
            client_loss = res.metrics.get("local_loss")
            client_param_norm = res.metrics.get("param_norm")
            client_delta_norm = res.metrics.get("delta_norm")
            client_n = res.num_examples

            # Log metrik individual per client sebelum agregasi (pembuktian FedAvg)
            if client_acc is not None:
                mlflow.log_metric(f"{cid}_train_accuracy", client_acc, step=rnd)
                accs.append(client_acc)
            if client_loss is not None:
                mlflow.log_metric(f"{cid}_train_loss", client_loss, step=rnd)
                losses.append(client_loss)
            if client_param_norm is not None:
                mlflow.log_metric(f"{cid}_param_norm", client_param_norm, step=rnd)
            if client_delta_norm is not None:
                mlflow.log_metric(f"{cid}_delta_norm", client_delta_norm, step=rnd)

            mlflow.log_metric(f"{cid}_fit_examples", client_n, step=rnd)
            num_examples_list.append(client_n)

        return accs, losses, num_examples_list

    def _log_client_eval_metrics(self, rnd, results):
        """Extract and log individual client evaluation metrics to MLflow before aggregation."""
        accs = []
        num_examples_list = []

        for idx, (client, res) in enumerate(results):
            cid = f"client_{idx + 1}"
            client_acc = res.metrics.get("accuracy")
            client_loss = res.loss
            client_n = res.num_examples

            # Log metrik individual per client sebelum agregasi (pembuktian FedAvg)
            if client_acc is not None:
                mlflow.log_metric(f"{cid}_val_accuracy", client_acc, step=rnd)
                accs.append(client_acc)
            if client_loss is not None:
                mlflow.log_metric(f"{cid}_val_loss", client_loss, step=rnd)

            mlflow.log_metric(f"{cid}_val_examples", client_n, step=rnd)
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

                # Cek dan simpan model terbaik berdasarkan akurasi tertinggi jika centralized tidak aktif
                if not self.evaluate_fn and weighted_acc > self.best_accuracy:
                    self.best_accuracy = weighted_acc
                    self.best_weights = self.current_weights
                    self.best_round = rnd
                    print(f"[ROUND {rnd}] New best global model achieved from client validation! Accuracy: {weighted_acc:.4f}")

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

                    # Jalankan evaluasi sentral khusus untuk BEST model pada server
                    if os.path.exists(config.SERVER_DATASET_PATH):
                        try:
                            _, X_test_c, _, y_test_c = dataset.load_data(config.SERVER_DATASET_PATH)
                            best_loss, best_acc = eval_utils.evaluate_and_log_metrics(
                                model_obj=lr_model,
                                X_test=X_test_c,
                                y_test=y_test_c,
                                artifact_path="best_model_centralized_evaluation",
                                metrics_prefix="best_model_centralized",
                                title_suffix="- Best Global Model (Centralized)",
                            )
                            print(f"[SERVER - BEST MODEL] Centralized evaluation of the BEST model completed: Loss = {best_loss:.4f}, Accuracy = {best_acc:.4f}")
                        except Exception as e:
                            print(f"[SERVER - BEST MODEL] Failed to run centralized evaluation on the best model: {e}")

                    # Daftarkan parameter model terbaik ke MLflow parent run
                    mlflow.log_params({"best_model_round": self.best_round, "best_model_val_accuracy": self.best_accuracy, "total_participating_clients": len(self.participant_clients)})

                    # Pengelolaan Model Registry tingkat lanjut menggunakan MlflowClient
                    self._register_best_model(model_name)
                else:
                    print("Warning: No weights found to reconstruct global model for MLflow logging.")

        return aggregated_result


def get_evaluate_fn(test_dataset_path, server_run_id):
    """Return an evaluation function for centralized evaluation on the server."""
    # Load dataset once
    _, X_test, _, y_test = dataset.load_data(test_dataset_path)

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config_dict: dict,
    ):
        # Reconstruct the model from parameters
        lr_model = model.get_lr_model()
        model_utils.set_model_parameters(lr_model, parameters)

        # Ensure active MLflow run is active on this thread context
        server_run = None
        if mlflow.active_run() is None:
            server_run = mlflow.start_run(run_id=server_run_id)

        try:
            loss, acc = eval_utils.evaluate_and_log_metrics(
                model_obj=lr_model,
                X_test=X_test,
                y_test=y_test,
                artifact_path=None,
                metrics_prefix="centralized",
                step=server_round,
                title_suffix=f"- Server Round {server_round}",
            )
        finally:
            if server_run is not None:
                mlflow.end_run()

        print(f"[SERVER] Centralized Evaluation for Round {server_round}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        return loss, {"accuracy": acc}

    return evaluate


def start_server():
    run_name = f"Federated_Server_Logistic_Regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        server_run_id = run.info.run_id
        mlflow.set_tag("mlflow.user", "server")
        mlflow.set_tag("handled_by", "Admin Server")
        mlflow.set_tag("project", "Federated Heart Disease")
        mlflow.set_tag("mlflow.note.content", "Eksperimen FL Murni: Server tidak memiliki akses ke dataset. Evaluasi hanya berdasarkan laporan metrik dari client.")

        # Log server configurations
        mlflow.log_params({"num_rounds": config.FL_NUM_ROUNDS, "strategy": "FedAvg"})

        def fit_config_with_run_id(server_round: int):
            config_dict = config.fit_config(server_round)
            config_dict["server_run_id"] = server_run_id
            return config_dict

        def evaluate_config(server_round: int):
            return {
                "server_round": server_round,
                "is_final_round": server_round == config.FL_NUM_ROUNDS,
                "server_run_id": server_run_id,
            }

        # Initialize the evaluate function if dataset path exists
        evaluate_fn = None
        if os.path.exists(config.SERVER_DATASET_PATH):
            try:
                evaluate_fn = get_evaluate_fn(config.SERVER_DATASET_PATH, server_run_id)
                print(f"[SERVER] Centralized evaluation enabled using dataset: {config.SERVER_DATASET_PATH}")
            except Exception as e:
                print(f"[SERVER] Failed to load centralized dataset for evaluation: {e}")
        else:
            print(f"[SERVER] Centralized evaluation dataset not found at: {config.SERVER_DATASET_PATH}")

        strategy = MLflowStrategy(
            on_fit_config_fn=fit_config_with_run_id,
            on_evaluate_config_fn=evaluate_config,
            evaluate_fn=evaluate_fn,
        )

        fl.server.start_server(server_address=config.FL_SERVER_ADDRESS, config=fl.server.ServerConfig(num_rounds=config.FL_NUM_ROUNDS), strategy=strategy)


if __name__ == "__main__":
    start_server()
