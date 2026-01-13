"""
Shared pytest fixtures for fairness_check tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fairness_check.config import EndpointConfig, DatasetConfig, FairnessConfig, Config


@pytest.fixture
def sample_y_pred():
    """Sample predictions for testing metrics."""
    return np.array([1, 0, 1, 0, 1, 1, 0, 0])


@pytest.fixture
def sample_y_true():
    """Sample true labels for testing metrics."""
    return np.array([1, 0, 1, 1, 1, 0, 0, 1])


@pytest.fixture
def sample_sensitive_features():
    """Sample sensitive attributes for testing metrics."""
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


@pytest.fixture
def sample_dataset_df():
    """Sample DataFrame for testing runner."""
    return pd.DataFrame(
        {
            "features": ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6"],
            "label": [1, 0, 1, 0, 1, 0],
            "sensitive_attribute": ["group_A", "group_A", "group_A", "group_B", "group_B", "group_B"],
        }
    )


@pytest.fixture
def perfect_fairness_data():
    """Data with perfect fairness (same rates across groups)."""
    return {
        "y_pred": np.array([1, 0, 1, 1, 0, 1]),
        "y_true": np.array([1, 0, 1, 1, 0, 1]),
        "sensitive": np.array(["A", "A", "A", "B", "B", "B"]),
    }


@pytest.fixture
def biased_data():
    """Data with maximum bias (one group always positive)."""
    return {
        "y_pred": np.array([1, 1, 1, 0, 0, 0]),
        "y_true": np.array([1, 0, 1, 0, 1, 0]),
        "sensitive": np.array(["A", "A", "A", "B", "B", "B"]),
    }


@pytest.fixture
def edge_case_empty():
    """Empty arrays for edge case testing."""
    return {"y_pred": np.array([]), "y_true": np.array([]), "sensitive": np.array([])}


@pytest.fixture
def edge_case_single_sample():
    """Single sample for edge case testing."""
    return {"y_pred": np.array([1]), "y_true": np.array([1]), "sensitive": np.array(["A"])}


@pytest.fixture
def edge_case_single_group():
    """All samples from single group."""
    return {
        "y_pred": np.array([1, 0, 1, 0]),
        "y_true": np.array([1, 0, 1, 1]),
        "sensitive": np.array(["A", "A", "A", "A"]),
    }


@pytest.fixture
def multiple_groups_data():
    """Data with 5 different groups."""
    return {
        "y_pred": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        "y_true": np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0]),
        "sensitive": np.array(["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]),
    }


@pytest.fixture
def endpoint_config():
    """Sample EndpointConfig for testing."""
    return EndpointConfig(
        url="http://test.com/classify", method="POST", headers={"Content-Type": "application/json"}, timeout=30
    )


@pytest.fixture
def endpoint_config_with_auth():
    """Sample EndpointConfig with authentication."""
    return EndpointConfig(
        url="http://test.com/classify",
        method="POST",
        headers={"Content-Type": "application/json"},
        timeout=30,
        auth_token="test-token-123",
    )


@pytest.fixture
def dataset_config(tmp_path):
    """Sample DatasetConfig for testing."""
    # Create a temporary CSV file
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {"features": ["feat1", "feat2", "feat3"], "label": [1, 0, 1], "sensitive_attribute": ["A", "B", "A"]}
    )
    df.to_csv(csv_path, index=False)

    return DatasetConfig(
        path=str(csv_path), features_column="features", labels_column="label", sensitive_column="sensitive_attribute"
    )


@pytest.fixture
def fairness_config():
    """Sample FairnessConfig for testing."""
    return FairnessConfig(demographic_parity_threshold=0.1, equal_opportunity_threshold=0.1)


@pytest.fixture
def full_config(endpoint_config, dataset_config, fairness_config):
    """Complete Config object for testing."""
    return Config(endpoint=endpoint_config, dataset=dataset_config, fairness=fairness_config)


@pytest.fixture
def temp_config_yaml(tmp_path):
    """Create a temporary config YAML file."""
    config_path = tmp_path / "config.yaml"
    config_content = """
endpoint:
  url: "http://localhost:8000/classify"
  method: "POST"
  timeout: 30
  headers:
    Content-Type: "application/json"

dataset:
  path: "data/test.csv"
  features_column: "features"
  labels_column: "label"
  sensitive_column: "sensitive_attribute"

fairness:
  demographic_parity_threshold: 0.1
  equal_opportunity_threshold: 0.1
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file with test data."""
    csv_path = tmp_path / "test_dataset.csv"
    df = pd.DataFrame(
        {
            "features": ["user1", "user2", "user3", "user4", "user5", "user6"],
            "label": [1, 0, 1, 0, 1, 0],
            "sensitive_attribute": ["male", "male", "male", "female", "female", "female"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_classifier_response_success():
    """Mock successful classifier response."""
    return {"prediction": 1}


@pytest.fixture
def mock_classifier_response_with_class():
    """Mock successful classifier response with 'class' field."""
    return {"class": 0}


@pytest.fixture
def mock_classifier_response_invalid():
    """Mock invalid classifier response (missing prediction/class)."""
    return {"result": 1, "confidence": 0.95}
