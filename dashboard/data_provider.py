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
    best_loss = 0.0
    current_round = 0

    for log in all_logs:
        msg = log["message"]
        if "client_" in log["container"]:
            active_clients_set.add(log["container"])

        cent_match = re.search(r"Centralized Evaluation (?:for )?Round (\d+): Loss = ([\d\.]+), Accuracy = ([\d\.]+)", msg)
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
                best_loss = loss

        client_eval_match = re.search(r"\[ROUND (\d+)\] New best global model achieved from client validation! Accuracy: ([\d\.]+)", msg)
        if client_eval_match:
            rnd = int(client_eval_match.group(1))
            acc = float(client_eval_match.group(2))
            current_round = max(current_round, rnd)
            fl_progress.append({"Round": rnd, "Metric": "Client Eval Accuracy", "Value": acc})
            if acc > best_accuracy:
                best_accuracy = acc
                best_round = rnd

    df_progress = pd.DataFrame(fl_progress) if fl_progress else pd.DataFrame(columns=["Round", "Metric", "Value"])

    if not df_progress.empty:
        df_pivot = df_progress.pivot_table(index="Round", columns="Metric", values="Value").reset_index()
    else:
        df_pivot = pd.DataFrame(columns=["Round", "Centralized Accuracy", "Centralized Loss", "Client Eval Accuracy"])

    health_states = {}
    now = datetime.now()
    for name in containers:
        recent_logs = [log for log in all_logs if log["container"] == name and (now - log["timestamp"]).total_seconds() < 300]  # noqa: PLR2004
        health_states[name] = len(recent_logs) > 0

    return {"current_round": current_round, "best_accuracy": best_accuracy, "best_round": best_round, "best_loss": best_loss, "active_clients_count": len(active_clients_set), "df_pivot": df_pivot, "health_states": health_states}


def get_mlflow_evaluation_artifacts():  # noqa: PLR0912, PLR0915
    """Retrieve best model evaluation reports and confusion matrices from MLflow."""
    results = {}
    try:
        requests.get("http://localhost:5000", timeout=0.5)
    except Exception:
        return {}

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()

        experiment = client.get_experiment_by_name("FL_Heart_Disease")
        if not experiment:
            return {}

        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attribute.start_time DESC"], max_results=10)
        if not runs:
            return {}

        server_run = None
        for run in runs:
            if "mlflow.parentRunId" not in run.data.tags:
                server_run = run
                break

        if not server_run:
            return {}

        client_runs = []
        for run in runs:
            if run.data.tags.get("mlflow.parentRunId") == server_run.info.run_id:
                client_runs.append(run)

        import os

        server_artifacts_local = {}
        local_path_base = os.path.join("mlruns", experiment.experiment_id, server_run.info.run_id, "artifacts", "best_model_centralized_evaluation")
        if os.path.isdir(local_path_base):
            for file in os.listdir(local_path_base):
                full_file_path = os.path.join(local_path_base, file)
                if file.endswith(".txt"):
                    with open(full_file_path, encoding="utf-8") as f:
                        server_artifacts_local["report"] = f.read()
                elif file.endswith(".png"):
                    server_artifacts_local["confusion_matrix"] = os.path.abspath(full_file_path)
        else:
            try:
                server_artifacts = client.list_artifacts(server_run.info.run_id, path="best_model_centralized_evaluation")
                for art in server_artifacts:
                    local_path = mlflow.artifacts.download_artifacts(run_id=server_run.info.run_id, artifact_path=art.path)
                    if art.path.endswith(".txt"):
                        with open(local_path, encoding="utf-8") as f:
                            server_artifacts_local["report"] = f.read()
                    elif art.path.endswith(".png"):
                        server_artifacts_local["confusion_matrix"] = local_path
            except Exception:
                pass

        if server_artifacts_local:
            results["server"] = server_artifacts_local

        for c_run in client_runs:
            c_id = c_run.data.tags.get("mlflow.user") or "unknown_client"
            client_artifacts_local = {}
            local_path_base = os.path.join("mlruns", experiment.experiment_id, c_run.info.run_id, "artifacts", "best_model_client_evaluation")
            if os.path.isdir(local_path_base):
                for file in os.listdir(local_path_base):
                    full_file_path = os.path.join(local_path_base, file)
                    if file.endswith(".txt"):
                        with open(full_file_path, encoding="utf-8") as f:
                            client_artifacts_local["report"] = f.read()
                    elif file.endswith(".png"):
                        client_artifacts_local["confusion_matrix"] = os.path.abspath(full_file_path)
            else:
                try:
                    client_artifacts = client.list_artifacts(c_run.info.run_id, path="best_model_client_evaluation")
                    for art in client_artifacts:
                        local_path = mlflow.artifacts.download_artifacts(run_id=c_run.info.run_id, artifact_path=art.path)
                        if art.path.endswith(".txt"):
                            with open(local_path, encoding="utf-8") as f:
                                client_artifacts_local["report"] = f.read()
                        elif art.path.endswith(".png"):
                            client_artifacts_local["confusion_matrix"] = local_path
                except Exception:
                    pass

            if client_artifacts_local:
                results[c_id] = client_artifacts_local

    except Exception as e:
        st.sidebar.warning(f"MLflow connection idle: {e}")
    return results


def fetch_metrics_from_mlflow():  # noqa: PLR0912, PLR0915, PLR0911
    """Retrieve training progression metrics history directly from MLflow.

    Returns:
        tuple: (df_pivot, best_round, best_acc, best_loss)
            - df_pivot: DataFrame with columns [Round, ...metric display names...]
            - best_round: round number of best centralized accuracy
            - best_acc: best centralized accuracy value
            - best_loss: loss value at best round
    """
    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    _EMPTY = pd.DataFrame(columns=["Round"]), 0, 0.0, 0.0

    try:
        requests.get("http://localhost:5000", timeout=0.5)
    except Exception:
        return _EMPTY

    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()

        experiment = client.get_experiment_by_name("FL_Heart_Disease")
        if not experiment:
            return _EMPTY
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=20,
        )
        if not runs:
            return _EMPTY

        server_run = None
        for run in runs:
            if "mlflow.parentRunId" not in run.data.tags:
                server_run = run
                break

        if not server_run:
            return _EMPTY

        run_id = server_run.info.run_id

        SERVER_METRICS = {
            "fedavg_train_accuracy": "FedAvg Accuracy",
            "fedavg_train_loss": "FedAvg Loss",
            "fedavg_val_accuracy": "FedAvg Eval Accuracy",
            "fedavg_val_loss": "FedAvg Eval Loss",
            "centralized_eval_accuracy": "Centralized Accuracy",
            "centralized_eval_loss": "Centralized Loss",
            "centralized_eval_precision": "Centralized Precision",
            "centralized_eval_recall": "Centralized Recall",
            "fedavg_fit_duration": "FedAvg Fit Duration (ms)",
            "max_fit_duration": "Max Fit Duration (ms)",
        }

        records = {}

        for mlflow_key, display_name in SERVER_METRICS.items():
            try:
                history = client.get_metric_history(run_id, mlflow_key)
                for m in history:
                    records.setdefault(m.step, {})[display_name] = m.value
            except Exception:
                pass

        client_runs = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == run_id]

        CLIENT_METRICS = {
            "train_accuracy": "Train Accuracy",
            "train_loss": "Train Loss",
            "train_precision": "Train Precision",
            "train_recall": "Train Recall",
            "val_accuracy": "Eval Accuracy",
            "val_loss": "Eval Loss",
            "val_precision": "Eval Precision",
            "val_recall": "Eval Recall",
            "param_norm": "Param Norm",
            "delta_norm": "Delta Norm",
            "fit_duration": "Fit Duration (ms)",
        }

        for c_run in client_runs:
            c_label = c_run.data.tags.get("mlflow.user") or c_run.data.tags.get("client_id") or "client"
            c_display = " ".join(p.capitalize() for p in c_label.split("_"))

            for mlflow_key, suffix in CLIENT_METRICS.items():
                try:
                    history = client.get_metric_history(c_run.info.run_id, mlflow_key)
                    display_name = f"{c_display} {suffix}"
                    for m in history:
                        records.setdefault(m.step, {})[display_name] = m.value
                except Exception:
                    pass

        df_list = [{"Round": step, **vals} for step, vals in records.items()]
        df_pivot = pd.DataFrame(df_list)

        if df_pivot.empty:
            return _EMPTY

        df_pivot = df_pivot.sort_values("Round").reset_index(drop=True)

        best_acc = 0.0
        best_round = 0
        best_loss = 0.0

        try:
            cent_hist = client.get_metric_history(run_id, "centralized_eval_accuracy")
            if cent_hist:
                best_acc = max(m.value for m in cent_hist)
                best_round = next(m.step for m in cent_hist if m.value == best_acc)

                cent_loss_hist = client.get_metric_history(run_id, "centralized_eval_loss")
                best_loss_m = next((m for m in cent_loss_hist if m.step == best_round), None)
                if best_loss_m:
                    best_loss = best_loss_m.value
        except Exception:
            pass

        if best_acc == 0.0:
            try:
                fed_hist = client.get_metric_history(run_id, "fedavg_val_accuracy")
                if fed_hist:
                    best_acc = max(m.value for m in fed_hist)
                    best_round = next(m.step for m in fed_hist if m.value == best_acc)

                    fed_loss_hist = client.get_metric_history(run_id, "fedavg_val_loss")
                    best_loss_m = next((m for m in fed_loss_hist if m.step == best_round), None)
                    if best_loss_m:
                        best_loss = best_loss_m.value
            except Exception:
                pass

        return df_pivot, best_round, best_acc, best_loss

    except Exception:
        return _EMPTY
