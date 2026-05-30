import json
import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # Headless mode for Docker
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss


def evaluate_and_log_metrics(  # noqa: PLR0913
    model_obj,
    X_test,
    y_test,
    artifact_path=None,
    metrics_prefix="eval",
    step=None,
    title_suffix="",
):
    """
    Evaluates the model on X_test, y_test.
    Generates and saves the classification report (txt, json) and confusion matrix plot.
    Logs them to MLflow active run along with loss and accuracy metrics.
    """
    # 1. Run predictions
    probs = model_obj.predict_proba(X_test)
    preds = model_obj.predict(X_test)

    from sklearn.metrics import f1_score, precision_score, recall_score

    # 2. Compute metrics
    loss = log_loss(y_test, probs, labels=[0, 1])
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    # 3. Log metrics to MLflow
    if mlflow.active_run() is not None:
        mlflow.log_metric(f"{metrics_prefix}_loss", loss, step=step)
        mlflow.log_metric(f"{metrics_prefix}_accuracy", acc, step=step)
        mlflow.log_metric(f"{metrics_prefix}_precision", precision, step=step)
        mlflow.log_metric(f"{metrics_prefix}_recall", recall, step=step)
        mlflow.log_metric(f"{metrics_prefix}_f1", f1, step=step)

    # 4. Generate and log artifacts ONLY if artifact_path is provided
    if artifact_path is not None:
        report_str = classification_report(y_test, preds, zero_division=0)
        report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds)

        with tempfile.TemporaryDirectory() as temp_dir:
            report_txt_path = os.path.join(temp_dir, "classification_report.txt")
            with open(report_txt_path, "w") as f:
                f.write(report_str)

            report_json_path = os.path.join(temp_dir, "classification_report.json")
            with open(report_json_path, "w") as f:
                json.dump(report_dict, f, indent=4)

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Healthy", "Heart Disease"],
                yticklabels=["Healthy", "Heart Disease"],
            )
            plt.title(f"Confusion Matrix {title_suffix}")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()

            cm_plot_path = os.path.join(temp_dir, "confusion_matrix.png")
            plt.savefig(cm_plot_path, dpi=100)
            plt.close()

            if mlflow.active_run() is not None:
                mlflow.log_artifact(report_txt_path, artifact_path=artifact_path)
                mlflow.log_artifact(report_json_path, artifact_path=artifact_path)
                mlflow.log_artifact(cm_plot_path, artifact_path=artifact_path)

    return loss, acc
