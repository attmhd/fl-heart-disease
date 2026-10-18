# Project Documentation

This project trains a binary heart disease classifier through federated learning. Flower coordinates a server and two clients, MLflow records experiments and registers models, and a Streamlit dashboard provides training controls and monitoring.

## Guides

1. [Architecture and workflow](architecture.md): services, communication, training rounds, and storage.
2. [Data and model](data-and-model.md): dataset preparation, loading, parameter exchange, and classifier configuration.
3. [Tracking and monitoring](tracking-and-monitoring.md): experiment runs, metrics, registry, dashboard, and logs.
4. [Setup and configuration](setup-and-configuration.md): dependencies, execution commands, addresses, and environment variables.

## Dataset roles

| Partition | Component | Role |
| --- | --- | --- |
| Hungary | `client_1` | Local training and evaluation |
| Switzerland | `client_2` | Local training and evaluation |
| Cleveland | `fl_server` | Centralized evaluation when the dataset can be loaded |

## Repository structure

```text
.
├── main.py                         # Streamlit entry point
├── dashboard/                      # Dashboard views, data access, and process control
├── src/
│   ├── client/client.py            # Flower client entry point
│   ├── server/server.py            # Flower server entry point
│   ├── configs/config.py           # Runtime configuration
│   ├── data/dataset.py             # CSV loading and train/evaluation split
│   ├── models/model.py             # Logistic Regression factory
│   └── utils/                      # Training, aggregation, evaluation, and parameter helpers
├── data/
│   ├── raw/                        # Raw dataset DVC pointer
│   └── processed/                  # Partitioned dataset DVC pointers
├── notebook/
│   └── eda_uci_heart_disease.ipynb   # Exploration and preprocessing
├── monitoring/
│   ├── alloy/                      # Docker log collection
│   ├── loki/                       # Log storage configuration
│   └── grafana/                    # Datasource and dashboard provisioning
├── tests/test_client.py             # Data loading and client tests
├── docs/                           # Project documentation
├── Dockerfile.server
├── Dockerfile.client
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Source modules

| Module | Responsibility |
| --- | --- |
| [`src/client/client.py`](../src/client/client.py) | Loads client data, creates the client, connects to Flower, and closes the active MLflow run on exit |
| [`src/server/server.py`](../src/server/server.py) | Creates the server run, builds the aggregation strategy, and starts the Flower gRPC server |
| [`src/configs/config.py`](../src/configs/config.py) | Reads environment variables and defines experiment names, dataset paths, and round configuration |
| [`src/data/dataset.py`](../src/data/dataset.py) | Reads CSV data, converts the target to binary labels, and creates training and evaluation splits |
| [`src/models/model.py`](../src/models/model.py) | Creates the Logistic Regression estimator from environment settings |
| [`src/utils/client_utils.py`](../src/utils/client_utils.py) | Implements `HeartDiseaseClient`, local training, local evaluation, and client metric logging |
| [`src/utils/server_utils.py`](../src/utils/server_utils.py) | Implements `MLflowStrategy`, aggregation metrics, model selection, centralized evaluation, and model registration |
| [`src/utils/model_utils.py`](../src/utils/model_utils.py) | Initializes estimators and reads or assigns model parameters |
| [`src/utils/eval_utils.py`](../src/utils/eval_utils.py) | Computes evaluation metrics and writes classification reports and confusion matrices |
| [`src/utils/data_utils.py`](../src/utils/data_utils.py) | Counts training samples and class frequencies |

The client and server modules also expose Flower `ClientApp` and `ServerApp` objects. Docker Compose uses their Python module entry points.

## Dashboard modules

| Module | Responsibility |
| --- | --- |
| `main.py` | Page layout, refresh controls, summary cards, and four dashboard tabs |
| `dashboard/tab_control_panel.py` | Training settings, CSV uploads, launch and stop buttons, and execution logs |
| `dashboard/tab_monitor.py` | Metric filters, Plotly charts, and evaluation artifacts |
| `dashboard/tab_health.py` | Container status cards and service topology display |
| `dashboard/tab_logs.py` | Container and severity filters for log messages |
| `dashboard/data_provider.py` | MLflow metric history, evaluation artifacts, and Loki queries |
| `dashboard/process_manager.py` | Docker commands, local process helpers, and container state lookup |
| `dashboard/commands.py` | Command definitions and local client environment settings |
| `dashboard/styles.py` | Dashboard styling and header rendering |

[Back to the project README](../README.md)
