import mlflow
from src.models import model
from src.utils import model_utils
from configs import config
import numpy as np
import flwr as fl
from datetime import datetime

# MLflow setup
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
mlflow.enable_system_metrics_logging()


class MLflowStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(self, rnd, results, failures):
        # 1. Agregasi bobot dari semua client
        aggregated_result = super().aggregate_fit(rnd, results, failures)

        if aggregated_result is not None:
            aggregated_weights, _ = aggregated_result

            with mlflow.start_run(run_name=f"round_{rnd}", nested=True):
                mlflow.set_tag("mlflow.user", "server")

                accs = []
                losses = []

                for _, res in results:
                    if "train_accuracy" in res.metrics:
                        accs.append(res.metrics["train_accuracy"])
                    if "local_loss" in res.metrics:
                        losses.append(res.metrics["local_loss"])

                if accs:
                    mlflow.log_metric("avg_train_accuracy", sum(accs)/len(accs))
                if losses:
                    mlflow.log_metric("avg_train_loss", sum(losses)/len(losses))

                # Reconstruct global model (hanya untuk logging/registry, tanpa data)
                ndarrays = fl.common.parameters_to_ndarrays(aggregated_weights)
                lr_model = model.get_lr_model()
                
                # Kita set hanya untuk memastikan model valid saat di-log
                # Karena tidak ada data di server, kita tidak memanggil fit() atau set_initial_params dengan data
                model_utils.set_model_parameters(lr_model, ndarrays)

                # Log Model & Model Registry (tanpa evaluasi terpusat)
                mlflow.sklearn.log_model(
                    sk_model=lr_model,
                    artifact_path="model",
                    registered_model_name="Heart_Disease_Global_Model"
                )

        return aggregated_result


def start_server():
    run_name = f"global_logreg_{config.FL_NUM_ROUNDS}r_{datetime.now().strftime('%Y-%m-%d')}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.user", "server")
        mlflow.set_tag("handled_by", "Admin Server")
        mlflow.set_tag("project", "Federated Heart Disease")
        mlflow.set_tag("mlflow.note.content", "Eksperimen FL Murni: Server tidak memiliki akses ke dataset. Evaluasi hanya berdasarkan laporan metrik dari client.")
        
        # Log server configurations
        mlflow.log_params({
            "num_rounds": config.FL_NUM_ROUNDS,
            "strategy": "FedAvg"
        })

        strategy = MLflowStrategy(on_fit_config_fn=config.fit_config)

        fl.server.start_server(
            server_address=config.FL_SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=config.FL_NUM_ROUNDS),
            strategy=strategy
        )


if __name__ == "__main__":
    start_server()