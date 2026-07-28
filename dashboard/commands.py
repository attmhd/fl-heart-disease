"""
Command definition settings for Docker and Local processes.

This module houses all CLI commands wrapped with `sg docker` security groups and
local Python execution environment variables mapping to preserve modularity and cleanliness.
"""

DOCKER_CMD_UP = "docker compose up -d --build"
DOCKER_CMD_STOP = "docker compose stop fl_server client_1 client_2"
DOCKER_CMD_PS = "docker compose ps --format json"

LOCAL_SERVER_CMD = ["src.server.server"]
LOCAL_CLIENT_CMD = ["src.client.client"]

LOCAL_CLIENT_1_ENV = {
    "CLIENT_ID": "client_1",
    "DATASET_PATH": "data/processed/heart_disease_hungary_cleaned.csv",
    "SERVER_ADDRESS": "127.0.0.1:8080",
}

LOCAL_CLIENT_2_ENV = {
    "CLIENT_ID": "client_2",
    "DATASET_PATH": "data/processed/heart_disease_switzerland_cleaned.csv",
    "SERVER_ADDRESS": "127.0.0.1:8080",
}
