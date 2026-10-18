# Tracking and Monitoring

[Documentation overview](README.md)

## MLflow experiments

The project records runs in the `FL_Heart_Disease` experiment. The server creates a parent run for the training session. Each client creates a run linked to the server through the `mlflow.parentRunId` tag received during training.

Run names contain the component name and a timestamp. Server and client runs use the tags `mlflow.user`, `handled_by`, and `project`.

The server logs `num_rounds` and the `FedAvg` strategy name. Client runs record the client ID, training sample count, class counts, and estimator parameters.

## Metrics

### Client training metrics

Each client records the following metrics after local training in every round:

| Metric | Meaning |
| --- | --- |
| `train_accuracy` | Accuracy on the local training split |
| `train_loss` | Log loss on the local training split |
| `train_precision` | Binary precision on the local training split |
| `train_recall` | Binary recall on the local training split |
| `val_accuracy` | Accuracy on the local evaluation split |
| `val_loss` | Log loss on the local evaluation split |
| `val_precision` | Binary precision on the local evaluation split |
| `val_recall` | Binary recall on the local evaluation split |
| `fit_duration` | Local estimator fit time in milliseconds |

These metrics describe the local model after fitting the parameters received for that round.

### Server metrics

| Metric | Meaning |
| --- | --- |
| `fedavg_train_accuracy` | Training-sample-weighted average of client training accuracy |
| `fedavg_train_loss` | Training-sample-weighted average of client training loss |
| `fedavg_val_accuracy` | Training-sample-weighted average of client local validation accuracy |
| `fedavg_val_loss` | Training-sample-weighted average of client local validation loss |
| `fedavg_fit_duration` | Unweighted mean of client fit durations in milliseconds |
| `max_fit_duration` | Maximum client fit duration in milliseconds |
| `centralized_eval_accuracy` | Global model accuracy on the Cleveland evaluation split |
| `centralized_eval_loss` | Global model log loss on the Cleveland evaluation split |
| `centralized_eval_precision` | Global model precision on the Cleveland evaluation split |
| `centralized_eval_recall` | Global model recall on the Cleveland evaluation split |
| `fedavg_eval_accuracy` | Evaluation-sample-weighted accuracy from final client evaluation of the selected global model |
| `fedavg_eval_loss` | Evaluation-sample-weighted loss from final client evaluation of the selected global model |
| `best_model_round` | Round associated with the model logged by the server |
| `best_model_val_accuracy` | Selection score retained by the strategy |

Centralized metrics are recorded when the server evaluator is active, including a baseline at round 0. Final client evaluation metrics are aggregated in the final round.

### Final client metrics

Clients record `best_model_client_accuracy`, `best_model_client_loss`, `best_model_client_precision`, and `best_model_client_recall` during final evaluation. Their metric step uses the selected model's round when supplied by the server.

## Evaluation artifacts

The evaluation helper writes three files for each evaluation artifact directory:

| File | Content |
| --- | --- |
| `classification_report.txt` | Text classification report |
| `classification_report.json` | Structured classification report, including per-class precision, recall, F1, and support |
| `confusion_matrix.png` | Confusion matrix image |

The server stores the selected model's centralized report under `best_model_centralized_evaluation` when Cleveland evaluation succeeds. Each evaluated client stores its final global-model report under `best_model_client_evaluation`.

## Model registry

The server reconstructs a Logistic Regression estimator from the selected parameter arrays and logs it with:

| Property | Value |
| --- | --- |
| Registered model name | `Heart_Disease_Global_Model` |
| Logged model name | `model` |
| Serialization format | `skops` |
| Registry aliases | `champion`, `Production` |

Registry processing updates the registered model description, finds the latest model version, and assigns version metadata. Version tags include `validation_accuracy`, `best_round`, `algorithm`, `environment`, and `total_participating_clients`.

Model logging is called from the final round's evaluation aggregation. When centralized evaluation of the selected model is performed after logging, its metrics are also associated with the logged model's `model_id`.

## Streamlit dashboard

[`main.py`](../main.py) provides summary cards for Best FL Round, Best Accuracy, and Best Loss, followed by four tabs.

| Tab | Functions |
| --- | --- |
| Training Control Panel | Configure training, upload processed CSV files, launch or stop training, and view execution output |
| Federated Training Monitor | Filter metrics, display round-by-round charts, and view classification reports and confusion matrices |
| Infrastructure Health | Display states for `fl_server`, `mlflow_server`, `client_1`, `client_2`, `loki`, and `alloy` |
| System Logs | View container logs and filter by container or severity keyword |

The control panel offers `saga`, `lbfgs`, `liblinear`, and `newton-cg` solvers. Its initial values are 5 rounds and one local maximum iteration. CSV uploads can replace the Cleveland, Hungary, or Switzerland processed file, with an application size limit of 1 MiB.

The launch button invokes `docker compose up -d --build`. The stop button invokes `docker compose stop fl_server client_1 client_2`. Command output is written to `local_fl_run.log`.

The dashboard reads metric history and artifacts from MLflow at `http://localhost:5000`. It selects the latest parent run from recent experiment runs and retrieves its associated client runs. If the metric table is empty, it uses metrics parsed from Loki logs. Summary cards use centralized accuracy history first, with local validation history as the fallback. Container states are read through `docker compose ps --format json`.

The sidebar provides a Loki endpoint field, an auto-refresh toggle, and a refresh interval from 2 to 15 seconds, initially set to 3 seconds.

## Container logs

Grafana Alloy discovers Docker containers through `/var/run/docker.sock` every 5 seconds. It assigns the labels `container`, `image`, and `container_id`, reads stdout and stderr, and forwards logs to Loki at `http://loki:3100/loki/api/v1/push`.

Loki uses filesystem storage and the `loki_data` volume. Grafana is provisioned with a Loki datasource and the dashboard definitions in `monitoring/grafana/dashboards/`:

- `docker-logs.json`
- `container-health.json`

The Streamlit dashboard queries Loki's `/loki/api/v1/query_range` endpoint. Its System Logs tab filters messages using container names and severity keywords.
