"""
Tests for fairness_check.config module.

Tests configuration loading, validation, and Pydantic models.
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from yaml.parser import ParserError

from fairness_check.config import (
    EndpointConfig,
    DatasetConfig,
    FairnessConfig,
    Config,
    load_config,
)


class TestEndpointConfig:
    """Tests for EndpointConfig Pydantic model."""

    def test_valid_config_all_fields(self):
        """Test valid endpoint config with all fields provided."""
        config = EndpointConfig(
            url="http://test.com/api",
            method="POST",
            headers={"Content-Type": "application/json"},
            timeout=60,
            auth_token="test-token-123"
        )

        assert config.url == "http://test.com/api"
        assert config.method == "POST"
        assert config.headers == {"Content-Type": "application/json"}
        assert config.timeout == 60
        assert config.auth_token == "test-token-123"

    def test_valid_config_with_defaults(self):
        """Test endpoint config with default values."""
        config = EndpointConfig(url="http://test.com/api")

        assert config.url == "http://test.com/api"
        assert config.method == "POST"  # Default
        assert config.headers == {}  # Default
        assert config.timeout == 30  # Default
        assert config.auth_token is None  # Default

    def test_method_normalization_lowercase_to_uppercase(self):
        """Test that HTTP method is normalized to uppercase."""
        config = EndpointConfig(url="http://test.com/api", method="post")
        assert config.method == "POST"

        config = EndpointConfig(url="http://test.com/api", method="get")
        assert config.method == "GET"

    def test_method_normalization_mixed_case(self):
        """Test method normalization with mixed case."""
        config = EndpointConfig(url="http://test.com/api", method="PoSt")
        assert config.method == "POST"

    def test_invalid_http_method(self):
        """Test that invalid HTTP method raises ValidationError."""
        with pytest.raises(ValidationError, match="Method must be GET or POST"):
            EndpointConfig(url="http://test.com/api", method="PUT")

    def test_invalid_http_method_delete(self):
        """Test that DELETE method is rejected."""
        with pytest.raises(ValidationError, match="Method must be GET or POST"):
            EndpointConfig(url="http://test.com/api", method="DELETE")

    def test_missing_required_url(self):
        """Test that missing URL raises ValidationError."""
        with pytest.raises(ValidationError):
            EndpointConfig()

    def test_auth_token_optional(self):
        """Test that auth_token is optional."""
        config = EndpointConfig(url="http://test.com/api")
        assert config.auth_token is None

    def test_custom_headers(self):
        """Test custom headers are stored correctly."""
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": "secret",
            "Authorization": "Bearer token"
        }
        config = EndpointConfig(url="http://test.com/api", headers=headers)
        assert config.headers == headers

    def test_timeout_custom_value(self):
        """Test custom timeout value."""
        config = EndpointConfig(url="http://test.com/api", timeout=120)
        assert config.timeout == 120


class TestDatasetConfig:
    """Tests for DatasetConfig Pydantic model."""

    def test_valid_config_all_fields(self):
        """Test valid dataset config with all fields provided."""
        config = DatasetConfig(
            path="/path/to/data.csv",
            features_column="input_features",
            labels_column="true_label",
            sensitive_column="protected_attribute"
        )

        assert config.path == "/path/to/data.csv"
        assert config.features_column == "input_features"
        assert config.labels_column == "true_label"
        assert config.sensitive_column == "protected_attribute"

    def test_valid_config_with_defaults(self):
        """Test dataset config with default column names."""
        config = DatasetConfig(path="/path/to/data.csv")

        assert config.path == "/path/to/data.csv"
        assert config.features_column == "features"  # Default
        assert config.labels_column == "label"  # Default
        assert config.sensitive_column == "sensitive_attribute"  # Default

    def test_missing_required_path(self):
        """Test that missing path raises ValidationError."""
        with pytest.raises(ValidationError):
            DatasetConfig()

    def test_custom_column_names(self):
        """Test custom column names are stored correctly."""
        config = DatasetConfig(
            path="data.csv",
            features_column="X",
            labels_column="y",
            sensitive_column="group"
        )
        assert config.features_column == "X"
        assert config.labels_column == "y"
        assert config.sensitive_column == "group"


class TestFairnessConfig:
    """Tests for FairnessConfig Pydantic model."""

    def test_valid_config_all_fields(self):
        """Test valid fairness config with all fields provided."""
        config = FairnessConfig(
            demographic_parity_threshold=0.05,
            equal_opportunity_threshold=0.08
        )

        assert config.demographic_parity_threshold == 0.05
        assert config.equal_opportunity_threshold == 0.08

    def test_valid_config_with_defaults(self):
        """Test fairness config with default threshold values."""
        config = FairnessConfig()

        assert config.demographic_parity_threshold == 0.1  # Default
        assert config.equal_opportunity_threshold == 0.1  # Default

    def test_zero_thresholds(self):
        """Test that zero thresholds are valid."""
        config = FairnessConfig(
            demographic_parity_threshold=0.0,
            equal_opportunity_threshold=0.0
        )
        assert config.demographic_parity_threshold == 0.0
        assert config.equal_opportunity_threshold == 0.0

    def test_threshold_as_integer(self):
        """Test that integer thresholds are converted to float."""
        config = FairnessConfig(
            demographic_parity_threshold=1,
            equal_opportunity_threshold=0
        )
        assert config.demographic_parity_threshold == 1.0
        assert config.equal_opportunity_threshold == 0.0
        assert isinstance(config.demographic_parity_threshold, float)

    def test_high_thresholds(self):
        """Test high threshold values."""
        config = FairnessConfig(
            demographic_parity_threshold=0.5,
            equal_opportunity_threshold=1.0
        )
        assert config.demographic_parity_threshold == 0.5
        assert config.equal_opportunity_threshold == 1.0


class TestConfig:
    """Tests for main Config Pydantic model."""

    def test_valid_complete_config(self, endpoint_config, dataset_config, fairness_config):
        """Test valid complete configuration."""
        config = Config(
            endpoint=endpoint_config,
            dataset=dataset_config,
            fairness=fairness_config
        )

        assert config.endpoint == endpoint_config
        assert config.dataset == dataset_config
        assert config.fairness == fairness_config

    def test_fairness_defaults_applied(self, endpoint_config, dataset_config):
        """Test that fairness config defaults are applied if not provided."""
        config = Config(
            endpoint=endpoint_config,
            dataset=dataset_config
        )

        assert config.fairness.demographic_parity_threshold == 0.1
        assert config.fairness.equal_opportunity_threshold == 0.1

    def test_missing_endpoint_section(self, dataset_config):
        """Test that missing endpoint section raises ValidationError."""
        with pytest.raises(ValidationError):
            Config(dataset=dataset_config)

    def test_missing_dataset_section(self, endpoint_config):
        """Test that missing dataset section raises ValidationError."""
        with pytest.raises(ValidationError):
            Config(endpoint=endpoint_config)

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'endpoint': {
                'url': 'http://test.com/api',
                'method': 'POST'
            },
            'dataset': {
                'path': 'data.csv'
            },
            'fairness': {
                'demographic_parity_threshold': 0.15,
                'equal_opportunity_threshold': 0.12
            }
        }

        config = Config(**config_dict)
        assert config.endpoint.url == 'http://test.com/api'
        assert config.dataset.path == 'data.csv'
        assert config.fairness.demographic_parity_threshold == 0.15


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_yaml(self, temp_config_yaml):
        """Test loading valid YAML configuration file."""
        config = load_config(temp_config_yaml)

        assert isinstance(config, Config)
        assert config.endpoint.url == "http://localhost:8000/classify"
        assert config.endpoint.method == "POST"
        assert config.endpoint.timeout == 30
        assert config.dataset.path == "data/test.csv"
        assert config.fairness.demographic_parity_threshold == 0.1

    def test_load_with_path_object(self, temp_config_yaml):
        """Test loading config using Path object."""
        config = load_config(Path(temp_config_yaml))

        assert isinstance(config, Config)
        assert config.endpoint.url == "http://localhost:8000/classify"

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        non_existent = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(non_existent)

    def test_yaml_syntax_error(self, tmp_path):
        """Test handling of YAML syntax errors."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("endpoint:\n  url: [this is: invalid yaml")

        with pytest.raises(ParserError, match="bad.yaml"):
            load_config(bad_yaml)

    def test_missing_required_field_endpoint(self, tmp_path):
        """Test that missing required endpoint field raises error."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'dataset': {
                'path': 'data.csv'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ParserError, match="Invalid configuration"):
            load_config(config_path)

    def test_missing_required_field_dataset(self, tmp_path):
        """Test that missing required dataset field raises error."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ParserError, match="Invalid configuration"):
            load_config(config_path)

    def test_missing_url_in_endpoint(self, tmp_path):
        """Test that missing URL in endpoint section raises error."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'method': 'POST'
            },
            'dataset': {
                'path': 'data.csv'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ParserError, match="Invalid configuration"):
            load_config(config_path)

    def test_invalid_http_method_in_yaml(self, tmp_path):
        """Test that invalid HTTP method in YAML raises error."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api',
                'method': 'DELETE'
            },
            'dataset': {
                'path': 'data.csv'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ParserError, match="Invalid configuration"):
            load_config(config_path)

    def test_extra_fields_ignored(self, tmp_path):
        """Test that extra fields in YAML are ignored by Pydantic."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api',
                'extra_field': 'should be ignored'
            },
            'dataset': {
                'path': 'data.csv',
                'unknown_field': 123
            },
            'unknown_section': {
                'foo': 'bar'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        # Should load successfully, extra fields ignored
        config = load_config(config_path)
        assert config.endpoint.url == 'http://test.com/api'
        assert config.dataset.path == 'data.csv'

    def test_invalid_type_for_timeout(self, tmp_path):
        """Test that invalid type for timeout raises error."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api',
                'timeout': 'not-a-number'
            },
            'dataset': {
                'path': 'data.csv'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ParserError, match="Invalid configuration"):
            load_config(config_path)

    def test_invalid_type_for_threshold(self, tmp_path):
        """Test that invalid type for fairness threshold raises error."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api'
            },
            'dataset': {
                'path': 'data.csv'
            },
            'fairness': {
                'demographic_parity_threshold': 'invalid'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ParserError, match="Invalid configuration"):
            load_config(config_path)

    def test_load_with_auth_token(self, tmp_path):
        """Test loading config with authentication token."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api',
                'auth_token': 'secret-token-123'
            },
            'dataset': {
                'path': 'data.csv'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)
        assert config.endpoint.auth_token == 'secret-token-123'

    def test_load_with_custom_headers(self, tmp_path):
        """Test loading config with custom headers."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api',
                'headers': {
                    'Content-Type': 'application/json',
                    'X-Custom-Header': 'value'
                }
            },
            'dataset': {
                'path': 'data.csv'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)
        assert config.endpoint.headers['Content-Type'] == 'application/json'
        assert config.endpoint.headers['X-Custom-Header'] == 'value'

    def test_load_minimal_config(self, tmp_path):
        """Test loading minimal valid configuration."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/api'
            },
            'dataset': {
                'path': 'data.csv'
            }
        }
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)

        # Check defaults are applied
        assert config.endpoint.method == "POST"
        assert config.endpoint.timeout == 30
        assert config.endpoint.headers == {}
        assert config.dataset.features_column == "features"
        assert config.fairness.demographic_parity_threshold == 0.1

    def test_empty_yaml_file(self, tmp_path):
        """Test that empty YAML file raises error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        with pytest.raises(ParserError, match="Invalid configuration"):
            load_config(config_path)
