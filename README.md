# Experiment Tracking and Model Registry on Heart Disease Classification using MLflow and Flower

A federated learning project for binary heart disease classification using Logistic Regression, Flower, and MLflow. Two clients train on the Hungary and Switzerland partitions of the UCI Heart Disease dataset. The server aggregates model parameters with FedAvg and uses the Cleveland partition for centralized evaluation when available.

The project includes a Streamlit dashboard for training control and monitoring, plus Grafana Alloy, Loki, and Grafana for container logs.

## Documentation

Start with the [documentation overview](docs/README.md).

| Guide | Contents |
| --- | --- |
| [Project overview](docs/README.md) | Components, repository structure, and module responsibilities |
| [Architecture and workflow](docs/architecture.md) | Services, communication, training rounds, and storage |
| [Data and model](docs/data-and-model.md) | Dataset partitions, preprocessing, data loading, and Logistic Regression |
| [Tracking and monitoring](docs/tracking-and-monitoring.md) | MLflow runs, metrics, model registry, dashboard, and logs |
| [Setup and configuration](docs/setup-and-configuration.md) | Installation, execution commands, service addresses, and environment variables |

## Main components

| Component | Implementation |
| --- | --- |
| Federated learning | Flower server and two clients using FedAvg |
| Classifier | Scikit-learn Logistic Regression |
| Experiment tracking | MLflow experiment `FL_Heart_Disease` |
| Model registry | `Heart_Disease_Global_Model` with `champion` and `Production` aliases |
| Training dashboard | Streamlit application in `main.py` |
| Log monitoring | Grafana Alloy, Loki, and Grafana |
| Dataset versioning | DVC with an S3-compatible remote |
| Runtime | Docker Compose and Python entry points |

## Author

Muhammad Attan — [2022610013.attan@itp.ac.id](mailto:2022610013.attan@itp.ac.id).
