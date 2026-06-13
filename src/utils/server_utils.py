# ruff: noqa
import logging
import os
import copy
import flwr as fl
import mlflow
from configs import config
from data import dataset
from models import model
from utils import eval_utils, model_utils

logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)


class MLflowStrategy(fl.server.strategy.FedAvg):
    def __init__(self, server_run_id: str, **kwargs):
        kwargs.setdefault("inplace", False)
        super().__init__(**kwargs)
        self.server_run_id = server_run_id
        self.best_accuracy = -1.0
        self.best_ndarrays = None
        self.best_round = None
        self.current_ndarrays = None
        self.participant_clients = set()

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        if server_round < config.FL_NUM_ROUNDS:
            return []
        if self.best_ndarrays is not None:
            print(f"[SERVER] Distributing BEST model (Round {self.best_round}, Accuracy: {self.best_accuracy:.4f}) to clients for final evaluation.")
            parameters_to_distribute = fl.common.ndarrays_to_parameters(self.best_ndarrays)
        else:
            parameters_to_distribute = parameters
        client_instructions = super().configure_evaluate(server_round, parameters_to_distribute, client_manager)
        if self.best_round is not None:
            for _, instruction in client_instructions:
                instruction.config["best_round"] = self.best_round
        return client_instructions

    def evaluate(self, server_round: int, parameters: fl.common.Parameters):
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is not None and server_round > 0:
            loss, metrics = eval_res
            centralized_acc = metrics.get("accuracy")
            if centralized_acc is not None and centralized_acc > self.best_accuracy:
                self.best_accuracy = centralized_acc
                self.best_ndarrays = copy.deepcopy(fl.common.parameters_to_ndarrays(parameters))
                self.best_round = server_round
                print(f"[SERVER - CENTRALIZED] New best model at Round {server_round}! Centralized Accuracy: {centralized_acc:.4f}")
        return eval_res

    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if aggregated_result is not None:
            aggregated_weights, _ = aggregated_result
            self.current_ndarrays = copy.deepcopy(fl.common.parameters_to_ndarrays(aggregated_weights))
            for client, _ in results:
                self.participant_clients.add(client.cid)

            server_run = None
            if mlflow.active_run() is None and self.server_run_id:
                server_run = mlflow.start_run(run_id=self.server_run_id)

            try:
                train_accs, train_losses, val_accs, val_losses, durations, num_examples_list = self._log_client_fit_metrics(rnd, results)

                def weighted_avg(values, counts):
                    if not values:
                        return None
                    total = sum(counts)
                    return sum(v * n for v, n in zip(values, counts)) / total if total > 0 else sum(values) / len(values)

                fedavg_train_acc = weighted_avg(train_accs, num_examples_list)
                fedavg_train_loss = weighted_avg(train_losses, num_examples_list)
                if fedavg_train_acc is not None:
                    mlflow.log_metric("fedavg_train_accuracy", fedavg_train_acc, step=rnd)
                if fedavg_train_loss is not None:
                    mlflow.log_metric("fedavg_train_loss", fedavg_train_loss, step=rnd)

                fedavg_val_acc = weighted_avg(val_accs, num_examples_list)
                fedavg_val_loss = weighted_avg(val_losses, num_examples_list)
                if fedavg_val_acc is not None:
                    mlflow.log_metric("fedavg_val_accuracy", fedavg_val_acc, step=rnd)
                if fedavg_val_loss is not None:
                    mlflow.log_metric("fedavg_val_loss", fedavg_val_loss, step=rnd)

                if durations:
                    mlflow.log_metric("fedavg_fit_duration", sum(durations) / len(durations), step=rnd)
                    mlflow.log_metric("max_fit_duration", max(durations), step=rnd)

                if not os.path.exists(config.SERVER_DATASET_PATH):
                    if rnd > 0 and fedavg_val_acc is not None and fedavg_val_acc > self.best_accuracy:
                        self.best_accuracy = fedavg_val_acc
                        self.best_ndarrays = copy.deepcopy(self.current_ndarrays)
                        self.best_round = rnd
                        print(f"[SERVER - DISTRIBUTED] New best model at Round {rnd}! FedAvg Val Accuracy: {fedavg_val_acc:.4f}")
            finally:
                if server_run is not None:
                    mlflow.end_run()
        return aggregated_result

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_result is not None:
            for client, _ in results:
                self.participant_clients.add(client.cid)

            server_run = None
            if mlflow.active_run() is None and self.server_run_id:
                server_run = mlflow.start_run(run_id=self.server_run_id)

            try:
                val_accs, val_losses, num_examples_list = [], [], []
                for client, res in results:
                    val_losses.append(res.loss)
                    num_examples_list.append(res.num_examples)
                    if res.metrics and "accuracy" in res.metrics:
                        val_accs.append(res.metrics["accuracy"])

                if num_examples_list:
                    total = sum(num_examples_list)
                    w_loss = sum(l * n for l, n in zip(val_losses, num_examples_list)) / total if total > 0 else sum(val_losses) / len(val_losses)
                    mlflow.log_metric("fedavg_eval_loss", w_loss, step=rnd)
                    if val_accs:
                        w_acc = sum(a * n for a, n in zip(val_accs, num_examples_list)) / total if total > 0 else sum(val_accs) / len(val_accs)
                        mlflow.log_metric("fedavg_eval_accuracy", w_acc, step=rnd)

                if rnd == config.FL_NUM_ROUNDS:
                    self._log_best_model_to_mlflow()
            finally:
                if server_run is not None:
                    mlflow.end_run()
        return aggregated_result

    def _log_best_model_to_mlflow(self):
        ndarrays_to_log = self.best_ndarrays if self.best_ndarrays is not None else self.current_ndarrays
        if ndarrays_to_log is None:
            print("[SERVER] Warning: No weights available to log to MLflow.")
            return

        best_rnd = self.best_round if self.best_round is not None else config.FL_NUM_ROUNDS
        print(f"[SERVER] Reconstructing best model from Round {best_rnd} (Accuracy: {self.best_accuracy:.4f}) for MLflow logging...")

        lr_model = model.get_lr_model()
        model_utils.set_model_parameters(lr_model, ndarrays_to_log)

        model_name = config.MLFLOW_MODEL_NAME
        model_info = mlflow.sklearn.log_model(
            sk_model=lr_model,
            name="model",
            registered_model_name=model_name,
            serialization_format="skops",
        )
        model_id = model_info.model_id if model_info is not None else None
        print(f"[SERVER] Best model (Round {best_rnd}) logged to MLflow. Model ID: {model_id}")

        if os.path.exists(config.SERVER_DATASET_PATH):
            try:
                _, X_test_c, _, y_test_c = dataset.load_data(config.SERVER_DATASET_PATH)
                dataset_name = os.path.basename(config.SERVER_DATASET_PATH) or "Unknown Dataset"
                best_loss, best_acc = eval_utils.evaluate_and_log_metrics(
                    model_obj=lr_model,
                    X_test=X_test_c,
                    y_test=y_test_c,
                    artifact_path="best_model_centralized_evaluation",
                    metrics_prefix="centralized",
                    step=best_rnd,
                    title_suffix=f"- Best Global Model (Centralized, Round {best_rnd})\nDataset: {dataset_name}",
                    model_id=model_id,
                )
                print(f"[SERVER - BEST MODEL] Centralized eval (Round {best_rnd}): Loss = {best_loss:.4f}, Accuracy = {best_acc:.4f}")
            except Exception as e:
                print(f"[SERVER - BEST MODEL] Centralized eval failed: {e}")

        mlflow.log_metrics(
            {
                "best_model_round": float(best_rnd),
                "best_model_val_accuracy": self.best_accuracy,
            }
        )
        mlflow.log_params(
            {
                "total_participating_clients": len(self.participant_clients),
            }
        )
        self._register_best_model(model_name)

    def _register_best_model(self, model_name):
        try:
            from mlflow import MlflowClient

            client = MlflowClient()
            client.update_registered_model(
                name=model_name,
                description="Model global hasil pembelajaran terfederasi menggunakan Flower untuk klasifikasi penyakit jantung.",
            )
            versions = client.search_model_versions(filter_string=f"name='{model_name}'", order_by=["version_number DESC"])
            if versions:
                latest_version = versions[0].version
                best_rnd = self.best_round if self.best_round is not None else config.FL_NUM_ROUNDS
                client.update_model_version(
                    name=model_name,
                    version=latest_version,
                    description=f"Model terbaik hasil agregasi federated learning — dipilih berdasarkan akurasi validasi tertinggi pada Ronde {best_rnd} dengan Akurasi: {self.best_accuracy:.4f}.",
                )
                client.set_model_version_tag(model_name, latest_version, "validation_accuracy", f"{self.best_accuracy:.4f}")
                client.set_model_version_tag(model_name, latest_version, "best_round", str(best_rnd))
                client.set_model_version_tag(model_name, latest_version, "algorithm", "Logistic Regression")
                client.set_model_version_tag(model_name, latest_version, "environment", "Production")
                client.set_model_version_tag(model_name, latest_version, "total_participating_clients", str(len(self.participant_clients)))
                client.set_registered_model_alias(model_name, "champion", latest_version)
                client.set_registered_model_alias(model_name, "Production", latest_version)
                print(f"[SERVER] Model Version {latest_version} promoted with alias '@Production'.")
        except Exception as e:
            print(f"[SERVER] Error during model registry operations: {e}")

    def _log_client_fit_metrics(self, rnd, results):
        train_accs, train_losses, val_accs, val_losses, durations, num_examples_list = [], [], [], [], [], []
        for idx, (client, res) in enumerate(results):
            n = res.num_examples
            m = res.metrics
            if m.get("train_accuracy") is not None:
                train_accs.append(m["train_accuracy"])
            if m.get("train_loss") is not None:
                train_losses.append(m["train_loss"])
            if m.get("val_accuracy") is not None:
                val_accs.append(m["val_accuracy"])
            if m.get("val_loss") is not None:
                val_losses.append(m["val_loss"])
            if m.get("fit_duration") is not None:
                durations.append(m["fit_duration"])
            num_examples_list.append(n)
        return train_accs, train_losses, val_accs, val_losses, durations, num_examples_list


def get_evaluate_fn(test_dataset_path, server_run_id):
    _, X_test, _, y_test = dataset.load_data(test_dataset_path)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config_dict: dict):
        if server_round == 0:
            print("[SERVER] Skipping centralized evaluation and logging at Round 0.")
            return None
        lr_model = model.get_lr_model()
        model_utils.set_model_parameters(lr_model, parameters)
        server_run = None
        if mlflow.active_run() is None:
            server_run = mlflow.start_run(run_id=server_run_id)
        try:
            dataset_name = os.path.basename(test_dataset_path) or "Unknown Dataset"
            loss, acc = eval_utils.evaluate_and_log_metrics(
                model_obj=lr_model,
                X_test=X_test,
                y_test=y_test,
                artifact_path=None,
                metrics_prefix="centralized",
                step=server_round,
                title_suffix=f"- Server Round {server_round}\nDataset: {dataset_name}",
            )
        finally:
            if server_run is not None:
                mlflow.end_run()
        print(f"[SERVER] Centralized Evaluation Round {server_round}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
        return loss, {"accuracy": acc}

    return evaluate


def build_strategy_with_mlflow(server_run_id: str):
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

    evaluate_fn = None
    if os.path.exists(config.SERVER_DATASET_PATH):
        try:
            evaluate_fn = get_evaluate_fn(config.SERVER_DATASET_PATH, server_run_id)
            print(f"[SERVER] Centralized evaluation enabled using dataset: {config.SERVER_DATASET_PATH}")
        except Exception as e:
            print(f"[SERVER] Failed to load centralized dataset for evaluation: {e}")
    else:
        print(f"[SERVER] Centralized evaluation dataset not found at: {config.SERVER_DATASET_PATH}")

    return MLflowStrategy(
        server_run_id=server_run_id,
        on_fit_config_fn=fit_config_with_run_id,
        on_evaluate_config_fn=evaluate_config,
        evaluate_fn=evaluate_fn,
        fraction_fit=config.FL_FRACTION_FIT,
        fraction_evaluate=config.FL_FRACTION_EVALUATE,
        min_fit_clients=config.FL_MIN_FIT_CLIENTS,
        min_evaluate_clients=config.FL_MIN_EVALUATE_CLIENTS,
    )
