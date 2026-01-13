"""
Integration tests for fairness_check module.

Tests end-to-end workflows with minimal mocking.
"""

import pytest
import pandas as pd
import yaml
from pathlib import Path
from unittest.mock import patch, Mock

from yaml.parser import ParserError

from fairness_check.config import load_config, Config, EndpointConfig, DatasetConfig, FairnessConfig
from fairness_check.runner import run_fairness_check
from fairness_check.ai_client import ClassifierClient


class TestEndToEndConfigToReport:
    """Test complete flow from config file to fairness report."""

    def test_full_workflow_with_perfect_fairness(self, tmp_path):
        """Test complete workflow with perfectly fair predictions."""
        # Create config file
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/classify',
                'method': 'POST',
                'timeout': 30
            },
            'dataset': {
                'path': str(tmp_path / "data.csv"),
                'features_column': 'features',
                'labels_column': 'label',
                'sensitive_column': 'sensitive_attribute'
            },
            'fairness': {
                'demographic_parity_threshold': 0.1,
                'equal_opportunity_threshold': 0.1
            }
        }
        config_path.write_text(yaml.dump(config_data))

        # Create dataset with perfect fairness
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'features': [f'user{i}' for i in range(20)],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2,
            'sensitive_attribute': ['A'] * 10 + ['B'] * 10
        })
        df.to_csv(csv_path, index=False)

        # Load config
        config = load_config(config_path)

        # Mock classifier with fair predictions (same rate for both groups)
        fair_predictions = [1, 0, 1, 0, 1] * 4  # 50% positive for both groups

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.side_effect = fair_predictions
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            # Run fairness check
            results = run_fairness_check(config, verbose=False)

            # Verify results
            assert results['total_predictions'] == 20
            assert 'accuracy' in results
            assert results['fairness_metrics']['demographic_parity_difference'] <= 0.1
            assert results['thresholds_met']['demographic_parity'] is True

    def test_full_workflow_with_biased_predictions(self, tmp_path):
        """Test complete workflow with biased predictions."""
        # Create config file
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/classify',
                'method': 'POST'
            },
            'dataset': {
                'path': str(tmp_path / "data.csv")
            },
            'fairness': {
                'demographic_parity_threshold': 0.1,
                'equal_opportunity_threshold': 0.1
            }
        }
        config_path.write_text(yaml.dump(config_data))

        # Create dataset
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'features': [f'user{i}' for i in range(20)],
            'label': [1, 0] * 10,
            'sensitive_attribute': ['GroupA'] * 10 + ['GroupB'] * 10
        })
        df.to_csv(csv_path, index=False)

        # Load config
        config = load_config(config_path)

        # Mock classifier with biased predictions (GroupA gets more positives)
        biased_predictions = [1] * 10 + [0] * 10  # 100% for A, 0% for B

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.side_effect = biased_predictions
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            # Run fairness check
            results = run_fairness_check(config, verbose=False)

            # Verify biased results
            assert results['total_predictions'] == 20
            assert results['fairness_metrics']['demographic_parity_difference'] > 0.5
            assert results['thresholds_met']['demographic_parity'] is False

    def test_full_workflow_with_authentication(self, tmp_path):
        """Test complete workflow with API authentication."""
        # Create config file with auth token
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/classify',
                'method': 'POST',
                'auth_token': 'secret-token-123',
                'headers': {
                    'Content-Type': 'application/json'
                }
            },
            'dataset': {
                'path': str(tmp_path / "data.csv")
            }
        }
        config_path.write_text(yaml.dump(config_data))

        # Create minimal dataset
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'features': ['user1', 'user2'],
            'label': [1, 0],
            'sensitive_attribute': ['A', 'B']
        })
        df.to_csv(csv_path, index=False)

        # Load config
        config = load_config(config_path)

        # Verify config loaded correctly
        assert config.endpoint.auth_token == 'secret-token-123'
        assert config.endpoint.headers['Content-Type'] == 'application/json'

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.return_value = 1
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            # Run fairness check
            results = run_fairness_check(config)

            # Verify ClassifierClient was initialized with correct config
            MockClient.assert_called_once()
            call_args = MockClient.call_args[0][0]
            assert call_args.auth_token == 'secret-token-123'

    def test_full_workflow_with_custom_thresholds(self, tmp_path):
        """Test complete workflow with custom fairness thresholds."""
        # Create config file with custom thresholds
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {
                'url': 'http://test.com/classify'
            },
            'dataset': {
                'path': str(tmp_path / "data.csv")
            },
            'fairness': {
                'demographic_parity_threshold': 0.3,  # More lenient
                'equal_opportunity_threshold': 0.25
            }
        }
        config_path.write_text(yaml.dump(config_data))

        # Create dataset
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'features': ['user1', 'user2', 'user3', 'user4'],
            'label': [1, 1, 0, 0],
            'sensitive_attribute': ['A', 'A', 'B', 'B']
        })
        df.to_csv(csv_path, index=False)

        # Load config
        config = load_config(config_path)

        # Mock with somewhat biased predictions
        predictions = [1, 1, 0, 1]  # A: 2/2=1.0, B: 1/2=0.5, diff=0.5

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.side_effect = predictions
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            # Run fairness check
            results = run_fairness_check(config)

            # With threshold of 0.3, DP diff of 0.5 should fail
            assert results['fairness_metrics']['demographic_parity_difference'] == pytest.approx(0.5)
            assert results['thresholds_met']['demographic_parity'] is False


class TestIntegrationWithRealComponents:
    """Test integration using real components where possible."""

    def test_config_loading_and_validation(self, tmp_path):
        """Test that config loading works with all validation."""
        config_path = tmp_path / "full_config.yaml"
        config_data = {
            'endpoint': {
                'url': 'https://api.example.com/v1/classify',
                'method': 'GET',
                'timeout': 60,
                'headers': {
                    'User-Agent': 'FairnessCheck/0.1.0',
                    'Accept': 'application/json'
                }
            },
            'dataset': {
                'path': '/data/test_dataset.csv',
                'features_column': 'input',
                'labels_column': 'output',
                'sensitive_column': 'protected_class'
            },
            'fairness': {
                'demographic_parity_threshold': 0.15,
                'equal_opportunity_threshold': 0.12
            }
        }
        config_path.write_text(yaml.dump(config_data))

        # Load and verify
        config = load_config(config_path)

        assert config.endpoint.url == 'https://api.example.com/v1/classify'
        assert config.endpoint.method == 'GET'
        assert config.endpoint.timeout == 60
        assert config.endpoint.headers['User-Agent'] == 'FairnessCheck/0.1.0'
        assert config.dataset.features_column == 'input'
        assert config.fairness.demographic_parity_threshold == 0.15

    def test_metrics_calculation_with_real_data(self, tmp_path):
        """Test metrics calculation with realistic data patterns."""
        # Create realistic dataset
        csv_path = tmp_path / "realistic_data.csv"
        df = pd.DataFrame({
            'features': [f'customer_{i}' for i in range(100)],
            'label': [1] * 60 + [0] * 40,  # 60% positive class
            'sensitive_attribute': ['male'] * 50 + ['female'] * 50
        })
        df.to_csv(csv_path, index=False)

        config = Config(
            endpoint=EndpointConfig(url='http://test.com/api'),
            dataset=DatasetConfig(path=str(csv_path)),
            fairness=FairnessConfig(
                demographic_parity_threshold=0.2,
                equal_opportunity_threshold=0.2
            )
        )

        # Create predictions with moderate bias
        # Male: 35/50 = 0.7, Female: 25/50 = 0.5, DP diff = 0.2
        predictions = [1] * 35 + [0] * 15 + [1] * 25 + [0] * 25

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.side_effect = predictions
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            results = run_fairness_check(config)

            assert results['total_predictions'] == 100
            assert 0.19 <= results['fairness_metrics']['demographic_parity_difference'] <= 0.21
            # With threshold of 0.2, should be right at the boundary


class TestIntegrationErrorScenarios:
    """Test error handling in integration scenarios."""

    def test_missing_dataset_file(self, tmp_path):
        """Test error when dataset file doesn't exist."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'endpoint': {'url': 'http://test.com/api'},
            'dataset': {'path': '/nonexistent/data.csv'}
        }
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)

        with pytest.raises(FileNotFoundError):
            run_fairness_check(config)

    def test_missing_column_in_dataset(self, tmp_path):
        """Test error when required column is missing from dataset."""
        csv_path = tmp_path / "incomplete.csv"
        df = pd.DataFrame({
            'features': ['a', 'b'],
            'label': [1, 0]
            # Missing 'sensitive_attribute' column
        })
        df.to_csv(csv_path, index=False)

        config = Config(
            endpoint=EndpointConfig(url='http://test.com/api'),
            dataset=DatasetConfig(path=str(csv_path)),
            fairness=FairnessConfig()
        )

        with pytest.raises(ValueError, match="Column 'sensitive_attribute' not found"):
            run_fairness_check(config)

    def test_invalid_yaml_config(self, tmp_path):
        """Test error with malformed YAML config."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("endpoint:\n  url: [invalid: yaml: syntax")

        with pytest.raises(ParserError, match="bad.yaml"):
            load_config(config_path)

    def test_api_connection_failure(self, tmp_path):
        """Test handling of API connection failures."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'features': ['user1', 'user2'],
            'label': [1, 0],
            'sensitive_attribute': ['A', 'B']
        })
        df.to_csv(csv_path, index=False)

        config = Config(
            endpoint=EndpointConfig(url='http://test.com/api'),
            dataset=DatasetConfig(path=str(csv_path)),
            fairness=FairnessConfig()
        )

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.side_effect = RuntimeError("Connection refused")
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(RuntimeError, match="Connection refused"):
                run_fairness_check(config)


class TestIntegrationMultipleGroups:
    """Test integration with multiple sensitive groups."""

    def test_three_sensitive_groups(self, tmp_path):
        """Test fairness evaluation with three demographic groups."""
        csv_path = tmp_path / "three_groups.csv"
        df = pd.DataFrame({
            'features': [f'person_{i}' for i in range(30)],
            'label': [1, 0] * 15,
            'sensitive_attribute': ['Asian'] * 10 + ['Black'] * 10 + ['White'] * 10
        })
        df.to_csv(csv_path, index=False)

        config = Config(
            endpoint=EndpointConfig(url='http://test.com/api'),
            dataset=DatasetConfig(path=str(csv_path)),
            fairness=FairnessConfig(demographic_parity_threshold=0.15)
        )

        # Create predictions with varying rates
        # Asian: 8/10=0.8, Black: 5/10=0.5, White: 3/10=0.3
        # Max diff = 0.8 - 0.3 = 0.5
        predictions = (
            [1] * 8 + [0] * 2 +     # Asian
            [1] * 5 + [0] * 5 +     # Black
            [1] * 3 + [0] * 7       # White
        )

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.side_effect = predictions
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            results = run_fairness_check(config)

            assert results['total_predictions'] == 30
            assert results['fairness_metrics']['demographic_parity_difference'] == pytest.approx(0.5)
            assert results['thresholds_met']['demographic_parity'] is False

    def test_five_sensitive_groups(self, tmp_path):
        """Test fairness evaluation with five demographic groups."""
        csv_path = tmp_path / "five_groups.csv"
        df = pd.DataFrame({
            'features': [f'item_{i}' for i in range(50)],
            'label': [1, 0] * 25,
            'sensitive_attribute': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 + ['E'] * 10
        })
        df.to_csv(csv_path, index=False)

        config = Config(
            endpoint=EndpointConfig(url='http://test.com/api'),
            dataset=DatasetConfig(path=str(csv_path)),
            fairness=FairnessConfig()
        )

        # Equal predictions across all groups (perfect fairness)
        predictions = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5  # 50% for each group

        with patch('fairness_check.runner.ClassifierClient') as MockClient:
            mock_client = Mock()
            mock_client.predict.side_effect = predictions
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            MockClient.return_value = mock_client

            results = run_fairness_check(config)

            assert results['total_predictions'] == 50
            assert results['fairness_metrics']['demographic_parity_difference'] == pytest.approx(0.0, abs=1e-9)
            assert results['thresholds_met']['demographic_parity'] is True
            assert results['thresholds_met']['equal_opportunity'] is True
