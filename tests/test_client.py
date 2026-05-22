import os
import sys
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.client.client import HeartDiseaseClient

# Import local modules using absolute path
from src.data import dataset


@pytest.fixture
def mock_csv_file():
    """Fixture to create a temporary mock CSV dataset with the correct schema."""
    # Column list
    columns = ["ca", "oldpeak", "thalch", "exang", "cp", "sex", "age", "slope", "chol", "fbs", "num"]

    # Generate mock data: 20 samples
    np.random.seed(42)
    data = {
        "ca": np.random.choice([0.0, 1.0, 2.0], size=20),
        "oldpeak": np.random.uniform(-2.5, 3.0, size=20),
        "thalch": np.random.uniform(-3.0, 3.0, size=20),
        "exang": np.random.choice([0, 1], size=20),
        "cp": np.random.choice([0, 1, 2, 3], size=20),
        "sex": np.random.choice([0, 1], size=20),
        "age": np.random.uniform(-3.0, 3.0, size=20),
        "slope": np.random.choice([1, 2, 3], size=20),
        "chol": np.random.uniform(-2.0, 4.0, size=20),
        "fbs": np.random.choice([0, 1], size=20),
        # Mix of 0 (healthy), 1, 2 (disease) to test binary mapping (num > 0)
        "num": np.random.choice([0, 1, 2], size=20),
    }

    df = pd.DataFrame(data, columns=columns)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)

    yield temp_path

    # Cleanup after test
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_load_data(mock_csv_file):
    """Test that load_data loads, maps classes to binary, and splits the data correctly."""
    X_train, X_test, y_train, y_test = dataset.load_data(mock_csv_file)

    # Check dimensions
    # Total samples is 20, 20% test size means 4 test, 16 train
    assert X_train.shape == (16, 10)
    assert X_test.shape == (4, 10)
    assert y_train.shape == (16,)
    assert y_test.shape == (4,)

    # Check that labels are binary (0 and 1 only)
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})

    # Check split deterministic reproducibility
    X_train2, X_test2, y_train2, y_test2 = dataset.load_data(mock_csv_file)
    np.testing.assert_array_equal(X_train, X_train2)
    np.testing.assert_array_equal(y_train, y_train2)


def test_heart_disease_client_init(mock_csv_file):
    """Test that the HeartDiseaseClient initializes correctly with local dataset."""
    X_train, X_test, y_train, y_test = dataset.load_data(mock_csv_file)
    client_id = "test_client_01"

    client = HeartDiseaseClient(X_train, y_train, X_test, y_test, client_id)

    assert client.client_id == client_id
    assert isinstance(client.learner, LogisticRegression)
    np.testing.assert_array_equal(client.X_train, X_train)
    np.testing.assert_array_equal(client.y_train, y_train)
    np.testing.assert_array_equal(client.X_test, X_test)
    np.testing.assert_array_equal(client.y_test, y_test)


def test_heart_disease_client_get_parameters(mock_csv_file):
    """Test get_parameters returns model parameters in proper format."""
    X_train, X_test, y_train, y_test = dataset.load_data(mock_csv_file)
    client = HeartDiseaseClient(X_train, y_train, X_test, y_test, "test_client_01")

    params = client.get_parameters(config={})

    assert isinstance(params, list)
    # Weights (coefficients) and intercepts are returned
    assert len(params) == 2
    assert params[0].shape == (1, 10)  # coefficients for 10 features
    assert params[1].shape == (1,)  # intercept


def test_heart_disease_client_fit(mock_csv_file):
    """Test the fit method trains locally, updates parameters and logs metrics."""
    X_train, X_test, y_train, y_test = dataset.load_data(mock_csv_file)
    client = HeartDiseaseClient(X_train, y_train, X_test, y_test, "test_client_01")

    initial_params = client.get_parameters(config={})

    # Mock Flower round configuration
    fit_config = {"server_round": "1"}

    local_weights, num_examples, metrics = client.fit(initial_params, fit_config)

    # Verify return types & lengths
    assert len(local_weights) == 2
    assert num_examples == 16
    assert isinstance(metrics, dict)

    # Check expected metrics are calculated and returned
    assert "train_accuracy" in metrics
    assert "local_loss" in metrics
    assert "param_norm" in metrics
    assert "delta_norm" in metrics

    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert metrics["local_loss"] >= 0.0
    assert metrics["param_norm"] >= 0.0
    assert metrics["delta_norm"] >= 0.0


def test_heart_disease_client_evaluate(mock_csv_file):
    """Test the evaluate method calculates loss and accuracy on test data."""
    X_train, X_test, y_train, y_test = dataset.load_data(mock_csv_file)
    client = HeartDiseaseClient(X_train, y_train, X_test, y_test, "test_client_01")

    params = client.get_parameters(config={})

    loss, num_examples, metrics = client.evaluate(params, config={})

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert num_examples == 4
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


# --- Real Dataset Validation Tests (Optional / Conditional) ---

REAL_DATASETS = ["data/processed/heart_disease_hungary_cleaned.csv", "data/processed/heart_disease_switzerland_cleaned.csv"]


@pytest.mark.parametrize("relative_path", REAL_DATASETS)
def test_real_dataset_integrity(relative_path):
    """If the real dataset CSV file exists in workspace, verify its existence, format, and loadability."""
    # Find absolute path from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    absolute_path = os.path.join(project_root, relative_path)

    if not os.path.exists(absolute_path):
        pytest.skip(f"Real dataset file {relative_path} not found in workspace.")

    # 1. Verify it is a valid file and not empty
    assert os.path.isfile(absolute_path), f"{relative_path} is not a valid file"
    assert os.path.getsize(absolute_path) > 0, f"{relative_path} is empty"

    # 2. Verify it has the correct CSV columns
    df = pd.read_csv(absolute_path)
    expected_columns = {"ca", "oldpeak", "thalch", "exang", "cp", "sex", "age", "slope", "chol", "fbs", "num"}
    assert expected_columns.issubset(df.columns), f"Missing expected columns in {relative_path}. Columns found: {df.columns.tolist()}"

    # 3. Verify it loads and splits properly
    X_train, X_test, y_train, y_test = dataset.load_data(absolute_path)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == 10
    assert X_test.shape[1] == 10

    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    # Ensure binary target mapping
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})
