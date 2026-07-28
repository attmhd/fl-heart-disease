import os
import subprocess
import sys

import streamlit as st

from .commands import (
    DOCKER_CMD_PS,
    DOCKER_CMD_STOP,
    DOCKER_CMD_UP,
    LOCAL_CLIENT_1_ENV,
    LOCAL_CLIENT_2_ENV,
    LOCAL_CLIENT_CMD,
    LOCAL_SERVER_CMD,
)


def init_session_states():
    """Initialize PIDs and state variables to persist across Streamlit reruns."""
    if "server_pid" not in st.session_state:
        st.session_state.server_pid = None
    if "client_pids" not in st.session_state:
        st.session_state.client_pids = []
    if "docker_active" not in st.session_state:
        st.session_state.docker_active = False


def is_process_running(pid):
    """Check if a local process is currently running using SIG 0 check."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_local_runs():
    """Kill all locally running server and client subprocesses using SIGTERM."""
    pids_to_kill = []
    if st.session_state.server_pid:
        pids_to_kill.append(st.session_state.server_pid)
    pids_to_kill.extend(st.session_state.client_pids)

    for pid in pids_to_kill:
        try:
            os.kill(pid, 15)  # SIGTERM
        except OSError:
            pass

    st.session_state.server_pid = None
    st.session_state.client_pids = []


def start_local_runs(rounds, solver, max_iter, c, penalty):
    """Launch local FL server and clients in background subprocesses."""
    stop_local_runs()

    env = os.environ.copy()
    env["FL_NUM_ROUNDS"] = str(rounds)
    env["MODEL_MAX_ITER"] = str(max_iter)
    env["MODEL_SOLVER"] = solver
    env["MODEL_C"] = str(c)
    env["MODEL_PENALTY"] = penalty
    env["PYTHONUNBUFFERED"] = "1"

    pwd = os.getcwd()
    env["PYTHONPATH"] = f"{pwd}:{pwd}/src"

    if "MLFLOW_TRACKING_URI" not in env:
        env["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    try:
        log_file = open("local_fl_run.log", "w", encoding="utf-8")
        log_file.write(f"=== Starting Local FL Run (Rounds: {rounds}, Solver: {solver}, Max Iter: {max_iter}, C: {c}, Penalty: {penalty}) ===\n")
        log_file.flush()
    except Exception as e:
        st.error(f"Failed to open log file: {e}")
        return False

    try:
        server_proc = subprocess.Popen([sys.executable, "-m"] + LOCAL_SERVER_CMD, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        st.session_state.server_pid = server_proc.pid
    except Exception as e:
        log_file.write(f"Failed to start server: {e}\n")
        log_file.close()
        return False

    try:
        env1 = env.copy()
        env1.update(LOCAL_CLIENT_1_ENV)
        client1_proc = subprocess.Popen([sys.executable, "-m"] + LOCAL_CLIENT_CMD, env=env1, stdout=log_file, stderr=subprocess.STDOUT)
        st.session_state.client_pids.append(client1_proc.pid)
    except Exception as e:
        log_file.write(f"Failed to start client_1: {e}\n")

    try:
        env2 = env.copy()
        env2.update(LOCAL_CLIENT_2_ENV)
        client2_proc = subprocess.Popen([sys.executable, "-m"] + LOCAL_CLIENT_CMD, env=env2, stdout=log_file, stderr=subprocess.STDOUT)
        st.session_state.client_pids.append(client2_proc.pid)
    except Exception as e:
        log_file.write(f"Failed to start client_2: {e}\n")

    log_file.close()
    return True


def start_docker_runs(rounds, solver, max_iter, c, penalty, fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=2, min_evaluate_clients=2):  # noqa: PLR0913
    """Launch Docker Compose FL services in background."""
    env = os.environ.copy()
    env["FL_NUM_ROUNDS"] = str(rounds)
    env["MODEL_MAX_ITER"] = str(max_iter)
    env["MODEL_SOLVER"] = solver
    env["MODEL_C"] = str(c)
    env["MODEL_PENALTY"] = penalty
    env["FL_FRACTION_FIT"] = str(fraction_fit)
    env["FL_FRACTION_EVALUATE"] = str(fraction_evaluate)
    env["FL_MIN_FIT_CLIENTS"] = str(min_fit_clients)
    env["FL_MIN_EVALUATE_CLIENTS"] = str(min_evaluate_clients)

    cmd = ["sg", "docker", "-c", DOCKER_CMD_UP]

    try:
        log_file = open("local_fl_run.log", "w", encoding="utf-8")
        log_file.write("=== Recreating Docker containers (fl_server, client_1, client_2) ===\n")
        log_file.write(f"Rounds: {rounds}, Solver: {solver}, Max Iter: {max_iter}, C: {c}, Penalty: {penalty}\n")
        log_file.write(f"Fraction Fit: {fraction_fit}, Fraction Eval: {fraction_evaluate}, Min Fit: {min_fit_clients}, Min Eval: {min_evaluate_clients}\n")
        log_file.write(f"Command executed: {' '.join(cmd)}\n")
        log_file.flush()

        subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        st.session_state.docker_active = True
    except Exception as e:
        st.error(f"Failed to trigger Docker Compose: {e}")
    finally:
        if "log_file" in locals() and not log_file.closed:
            log_file.close()


def stop_docker_runs():
    """Stop Docker Compose FL services in background."""
    env = os.environ.copy()
    cmd = ["sg", "docker", "-c", DOCKER_CMD_STOP]

    try:
        log_file = open("local_fl_run.log", "a", encoding="utf-8")
        log_file.write("\n=== Terminating Docker FL services ===\n")
        log_file.write(f"Command executed: {' '.join(cmd)}\n")
        log_file.flush()

        subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)  # noqa: PLW1510
        st.session_state.docker_active = False
    except Exception as e:
        st.error(f"Failed to stop Docker services: {e}")
    finally:
        if "log_file" in locals() and not log_file.closed:
            log_file.close()


def get_docker_health_states(containers):
    """Retrieve actual running states of Docker containers using docker compose ps."""
    states = {name: False for name in containers}
    try:
        cmd = ["sg", "docker", "-c", DOCKER_CMD_PS]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=False)  # noqa: PLW1510
        if res.returncode == 0 and res.stdout.strip():
            import json

            output = res.stdout.strip()
            if output.startswith("["):
                services = json.loads(output)
            else:
                services = [json.loads(line) for line in output.split("\n") if line.strip()]

            for svc in services:
                name = svc.get("Service") or svc.get("Name")
                state = (svc.get("State") or svc.get("Status") or "").lower()
                if name in states:
                    states[name] = "running" in state or "up" in state or "healthy" in state
    except Exception:
        pass
    return states
