# ruff: noqa: E402
import os
import warnings

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
# Suppress MLflow requirements/mismatch and future deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

from datetime import datetime

import flwr as fl
import mlflow

from configs import config
from utils.server_utils import build_strategy_with_mlflow

# MLflow setup
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    """ServerApp factory function — called by flower-superexec / flower-superlink.

    MLflow parent run is started here so it wraps the full training session.
    """
    run_name = f"Federated_Server_Logistic_Regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    active = mlflow.active_run()
    run = active if active else mlflow.start_run(run_name=run_name)
    server_run_id = run.info.run_id

    mlflow.set_tag("mlflow.user", "server")
    mlflow.set_tag("handled_by", "Admin Server")
    mlflow.set_tag("project", "Federated Heart Disease")
    mlflow.set_tag(
        "mlflow.note.content",
        "Eksperimen FL Murni: Server tidak memiliki akses ke dataset. Evaluasi hanya berdasarkan laporan metrik dari client.",
    )
    mlflow.log_params({"num_rounds": config.FL_NUM_ROUNDS, "strategy": "FedAvg"})

    strategy = build_strategy_with_mlflow(server_run_id)
    server_config = fl.server.ServerConfig(num_rounds=config.FL_NUM_ROUNDS)
    return fl.server.ServerAppComponents(strategy=strategy, config=server_config)


# Module-level ServerApp — loaded by flower-superexec (plugin-type=serverapp)
app = fl.server.ServerApp(server_fn=server_fn)


def start_server():
    """Legacy entry point — starts server directly via Python (for backward compatibility).

    Wraps the MLflow parent run and launches the gRPC infrastructure inline.
    Kept for local/Docker Compose runs where flower-superlink is not separately deployed.
    """
    run_name = f"Federated_Server_Logistic_Regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        server_run_id = run.info.run_id
        mlflow.set_tag("mlflow.user", "server")
        mlflow.set_tag("handled_by", "Admin Server")
        mlflow.set_tag("project", "Federated Heart Disease")
        mlflow.set_tag(
            "mlflow.note.content",
            "Eksperimen FL Murni: Server tidak memiliki akses ke dataset. Evaluasi hanya berdasarkan laporan metrik dari client.",
        )
        mlflow.log_params({"num_rounds": config.FL_NUM_ROUNDS, "strategy": "FedAvg"})

        strategy = build_strategy_with_mlflow(server_run_id)

        # Use the new non-deprecated gRPC server path directly
        from flwr.server.compat.app import init_defaults, run_fl
        from flwr.server.superlink.fleet.grpc_bidi.grpc_server import start_grpc_server

        initialized_server, initialized_config = init_defaults(
            server=None,
            config=fl.server.ServerConfig(num_rounds=config.FL_NUM_ROUNDS),
            strategy=strategy,
            client_manager=None,
        )

        print(f"[SERVER] Starting gRPC server on {config.FL_SERVER_ADDRESS}")
        grpc_server = start_grpc_server(
            client_manager=initialized_server.client_manager(),
            server_address=config.FL_SERVER_ADDRESS,
        )

        try:
            run_fl(server=initialized_server, config=initialized_config)
        finally:
            grpc_server.stop(grace=1)


if __name__ == "__main__":
    start_server()
