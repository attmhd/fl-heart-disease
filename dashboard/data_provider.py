import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


def fetch_loki_logs(loki_url, query, limit=500):
    """Query logs from Loki server."""
    try:
        url = f"{loki_url}/loki/api/v1/query_range"
        params = {"query": query, "limit": limit, "direction": "BACKWARD"}
        response = requests.get(url, params=params, timeout=2.0)
        if response.status_code == 200:  # noqa: PLR2004
            data = response.json()
            logs = []
            if data.get("status") == "success":
                results = data.get("data", {}).get("result", [])
                for result in results:
                    container = result.get("stream", {}).get("container", "unknown")
                    for val in result.get("values", []):
                        timestamp = datetime.fromtimestamp(int(val[0]) / 1e9)
                        logs.append({"timestamp": timestamp, "container": container, "message": val[1].strip()})
            logs = sorted(logs, key=lambda x: x["timestamp"])
            return logs
    except Exception:
        pass
    return []


def parse_metrics_and_progress(all_logs, containers):
    """Parse Loki logs to extract metrics, training progress, and container health states."""
    fl_progress = []
    active_clients_set = set()
    best_accuracy = 0.0
    best_round = 0
    current_round = 0

    for log in all_logs:
        msg = log["message"]
        if "client_" in log["container"]:
            active_clients_set.add(log["container"])

        # Pattern 1: Centralized Evaluation
        cent_match = re.search(r"Centralized Evaluation for Round (\d+): Loss = ([\d\.]+), Accuracy = ([\d\.]+)", msg)
        if cent_match:
            rnd = int(cent_match.group(1))
            loss = float(cent_match.group(2))
            acc = float(cent_match.group(3))
            current_round = max(current_round, rnd)
            fl_progress.append({"Round": rnd, "Metric": "Centralized Accuracy", "Value": acc})
            fl_progress.append({"Round": rnd, "Metric": "Centralized Loss", "Value": loss})
            if acc > best_accuracy:
                best_accuracy = acc
                best_round = rnd

        # Pattern 2: Global Validation via Client
        client_eval_match = re.search(r"\[ROUND (\d+)\] New best global model achieved from client validation! Accuracy: ([\d\.]+)", msg)
        if client_eval_match:
            rnd = int(client_eval_match.group(1))
            acc = float(client_eval_match.group(2))
            current_round = max(current_round, rnd)
            fl_progress.append({"Round": rnd, "Metric": "Client Val Accuracy", "Value": acc})
            if acc > best_accuracy:
                best_accuracy = acc
                best_round = rnd

    df_progress = pd.DataFrame(fl_progress) if fl_progress else pd.DataFrame(columns=["Round", "Metric", "Value"])

    if not df_progress.empty:
        df_pivot = df_progress.pivot_table(index="Round", columns="Metric", values="Value").reset_index()
    else:
        df_pivot = pd.DataFrame(columns=["Round", "Centralized Accuracy", "Centralized Loss", "Client Val Accuracy"])

    # Determine container health states
    health_states = {}
    now = datetime.now()
    for name in containers:
        # Check if the container sent logs within the last 5 minutes (300 seconds) to ensure real-time liveness
        recent_logs = [log for log in all_logs if log["container"] == name and (now - log["timestamp"]).total_seconds() < 300]  # noqa: PLR2004
        health_states[name] = len(recent_logs) > 0

    return {"current_round": current_round, "best_accuracy": best_accuracy, "best_round": best_round, "active_clients_count": len(active_clients_set), "df_pivot": df_pivot, "health_states": health_states}


def get_mlflow_evaluation_artifacts():  # noqa: PLR0912
    """Retrieve best model evaluation reports and confusion matrices from MLflow."""
    results = {}
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # Set tracking URI to match the system default
        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()

        # Find active experiment
        experiment = client.get_experiment_by_name("FL_Heart_Disease")
        if not experiment:
            return {}

        # Get all runs sorted by start_time descending
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attribute.start_time DESC"], max_results=10)
        if not runs:
            return {}

        # Find the latest parent run (the server run)
        server_run = None
        for run in runs:
            # Server runs don't have mlflow.parentRunId in tags
            if "mlflow.parentRunId" not in run.data.tags:
                server_run = run
                break

        if not server_run:
            return {}

        # Find corresponding client runs (child runs)
        client_runs = []
        for run in runs:
            if run.data.tags.get("mlflow.parentRunId") == server_run.info.run_id:
                client_runs.append(run)

        # 1. Fetch Server Centralized Evaluation artifacts
        server_artifacts_local = {}
        server_artifacts = client.list_artifacts(server_run.info.run_id, path="best_model_centralized_evaluation")
        for art in server_artifacts:
            local_path = mlflow.artifacts.download_artifacts(run_id=server_run.info.run_id, artifact_path=art.path)
            # Read contents
            if art.path.endswith(".txt"):
                with open(local_path) as f:
                    server_artifacts_local["report"] = f.read()
            elif art.path.endswith(".png"):
                server_artifacts_local["confusion_matrix"] = local_path

        if server_artifacts_local:
            results["server"] = server_artifacts_local

        # 2. Fetch Client evaluation artifacts
        for c_run in client_runs:
            c_id = c_run.data.tags.get("mlflow.user") or "unknown_client"
            client_artifacts_local = {}
            client_artifacts = client.list_artifacts(c_run.info.run_id, path="best_model_client_evaluation")
            for art in client_artifacts:
                local_path = mlflow.artifacts.download_artifacts(run_id=c_run.info.run_id, artifact_path=art.path)
                if art.path.endswith(".txt"):
                    with open(local_path) as f:
                        client_artifacts_local["report"] = f.read()
                elif art.path.endswith(".png"):
                    client_artifacts_local["confusion_matrix"] = local_path

            if client_artifacts_local:
                results[c_id] = client_artifacts_local

    except Exception as e:
        st.sidebar.warning(f"MLflow connection idle: {e}")
    return results
