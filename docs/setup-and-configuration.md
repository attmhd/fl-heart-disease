# Setup and Configuration

[Documentation overview](README.md)

Run the commands below from the repository root.

## Requirements

- Python 3.11, matching `.python-version` and the Docker base images.
- uv for Python dependency management.
- Docker Engine and Docker Compose v2 for the container stack.
- Access to the project's DVC remote to retrieve versioned CSV files.
- For dashboard Docker controls: a Linux environment with `sg` and access to the `docker` group.

## Install dependencies and retrieve data

```bash
uv sync --locked
uv run dvc pull
```

DVC reads the remote configuration from `.dvc/config`. Authentication uses the local DVC/S3 configuration for that remote.

The training stack reads these processed files:

```text
data/processed/heart_disease_hungary_cleaned.csv
data/processed/heart_disease_switzerland_cleaned.csv
data/processed/heart_disease_cleveland_cleaned.csv
```

The preprocessing notebook also exports these filenames. Its output files can be placed in `data/processed/` for runtime loading.

## Start the Docker stack

```bash
docker compose up -d --build
```

This starts MLflow, the Flower server, both clients, Loki, Alloy, and Grafana.

Follow training logs:

```bash
docker compose logs -f fl_server client_1 client_2
```

Display service status:

```bash
docker compose ps
```

Stop the training services:

```bash
docker compose stop fl_server client_1 client_2
```

Stop all services:

```bash
docker compose stop
```

## Start the Streamlit dashboard

The dashboard entry point is `main.py`. Its `streamlit` and `plotly` imports can be supplied through uv's additional runtime dependencies:

```bash
uv run --with streamlit --with plotly streamlit run main.py
```

Open `http://localhost:8501`. Use the Training Control Panel to configure and launch the Compose stack. The dashboard reads MLflow at `http://localhost:5000` and initially queries Loki at `http://localhost:3100`.

## Run the training processes locally

The server and clients can also run as Python processes. Each command below runs in a separate terminal from the repository root. Start this mode separately from the Docker training services because both use port `8080`.

Start MLflow:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Start the Flower server:

```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run python -m src.server.server
```

Start client 1:

```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  SERVER_ADDRESS=127.0.0.1:8080 CLIENT_ID=client_1 \
  DATASET_PATH=data/processed/heart_disease_hungary_cleaned.csv \
  uv run python -m src.client.client
```

Start client 2:

```bash
PYTHONPATH=src MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  SERVER_ADDRESS=127.0.0.1:8080 CLIENT_ID=client_2 \
  DATASET_PATH=data/processed/heart_disease_switzerland_cleaned.csv \
  uv run python -m src.client.client
```

The local Python processes use the same dataset loader, model factory, client implementation, and server strategy as the Docker entry points.

## Service addresses

| Component | Host address | Container address |
| --- | --- | --- |
| Streamlit | `http://localhost:8501` | Runs on the host |
| MLflow | `http://localhost:5000` | `http://mlflow_server:5000` |
| Flower gRPC server | `localhost:8080` | `fl_server:8080` |
| Loki | `http://localhost:3100` | `http://loki:3100` |
| Grafana | `http://localhost:3000` | `http://grafana:3000` |

Grafana's initial administrator credentials are `admin` / `admin`.

## Environment variables

### Federation

| Variable | Code default | Description |
| --- | --- | --- |
| `FL_NUM_ROUNDS` | `10` | Number of training rounds |
| `FL_FRACTION_FIT` | `1.0` | Fraction of clients sampled for local training |
| `FL_FRACTION_EVALUATE` | `1.0` | Fraction of clients sampled for final evaluation |
| `FL_MIN_FIT_CLIENTS` | `2` | Minimum number of clients selected for training |
| `FL_MIN_EVALUATE_CLIENTS` | `2` | Minimum number of clients selected for evaluation |

Compose forwards these environment variables to `fl_server`.

### Model

| Variable | Code default | Description |
| --- | --- | --- |
| `MODEL_MAX_ITER` | `10` | Maximum solver iterations per local fit |
| `MODEL_SOLVER` | `saga` | Logistic Regression solver |
| `MODEL_C` | `1.0` | Inverse regularization strength |
| `MODEL_PENALTY` | `l2` | Regularization setting: `l2`, `l1`, or `none` |

Compose forwards model variables to the server and both clients. The dashboard supplies its form values, initially 5 rounds and `MODEL_MAX_ITER=1`.

### Connections and dataset paths

| Variable | Code default | Compose value |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | Read from the environment | `http://mlflow_server:5000` |
| `SERVER_DATASET_PATH` | `data/processed/heart_disease_cleveland_cleaned.csv` | `/app/data/processed/heart_disease_cleveland_cleaned.csv` |
| `SERVER_ADDRESS` | `fl_server:8080` | `fl_server:8080` for both clients |
| `DATASET_PATH` | Read from the environment | Hungary for client 1; Switzerland for client 2, under `/app/data/processed/` |
| `CLIENT_ID` | `client_unknown` | `client_1` or `client_2` |

The server binds to `0.0.0.0:8080`. Experiment and registered model names are defined in `src/configs/config.py`.

For numeric FL and model settings, missing, empty, or nonnumeric environment values use the code defaults. Dataset paths and connection addresses are assigned directly in `docker-compose.yml` for container execution.

## Existing tests

```bash
PYTHONPATH=src uv run pytest
```

`tests/test_client.py` contains tests for CSV loading, binary target conversion, deterministic splitting, client initialization, parameter shapes, local fit, and evaluation. It also includes integrity checks for the Hungary and Switzerland CSV files; these checks are skipped when the files are unavailable.
