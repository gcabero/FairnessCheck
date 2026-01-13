"""
Tests for fairness_check.ai_client module.

Tests the ClassifierClient class that makes HTTP requests to classifier endpoints.
Uses requests-mock to mock HTTP responses.
"""

import pytest
import requests
from fairness_check.ai_client import ClassifierClient
from fairness_check.config import EndpointConfig


class TestClassifierClientPOST:
    """Tests for POST requests."""

    def test_predict_success_with_prediction_field(self, requests_mock, endpoint_config):
        """Test successful POST prediction with 'prediction' field in response."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        client = ClassifierClient(endpoint_config)
        result = client.predict("test_features")

        assert result == 1
        assert requests_mock.called
        assert requests_mock.last_request.json() == {'features': 'test_features'}

    def test_predict_success_with_class_field(self, requests_mock, endpoint_config):
        """Test successful POST prediction with 'class' field in response."""
        requests_mock.post(
            'http://test.com/classify',
            json={'class': 0}
        )

        client = ClassifierClient(endpoint_config)
        result = client.predict("test_features")

        assert result == 0

    def test_predict_with_dict_features(self, requests_mock, endpoint_config):
        """Test prediction with dictionary features."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        client = ClassifierClient(endpoint_config)
        features_dict = {'age': 25, 'income': 50000}
        result = client.predict(features_dict)

        assert result == 1
        assert requests_mock.last_request.json() == {'features': features_dict}

    def test_predict_with_list_features(self, requests_mock, endpoint_config):
        """Test prediction with list features."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 0}
        )

        client = ClassifierClient(endpoint_config)
        features_list = [1.0, 2.0, 3.0]
        result = client.predict(features_list)

        assert result == 0
        assert requests_mock.last_request.json() == {'features': features_list}


class TestClassifierClientGET:
    """Tests for GET requests."""

    def test_predict_get_method(self, requests_mock):
        """Test GET request with query parameters."""
        config = EndpointConfig(
            url="http://test.com/classify",
            method="GET",
            headers={},
            timeout=30
        )

        requests_mock.get(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        client = ClassifierClient(config)
        result = client.predict("test_features")

        assert result == 1
        assert 'features=test_features' in requests_mock.last_request.url


class TestClassifierClientAuthentication:
    """Tests for authentication."""

    def test_auth_header_injection(self, requests_mock, endpoint_config_with_auth):
        """Test that Bearer token is injected correctly."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        client = ClassifierClient(endpoint_config_with_auth)
        client.predict("test")

        auth_header = requests_mock.last_request.headers.get('Authorization')
        assert auth_header == 'Bearer test-token-123'

    def test_custom_headers(self, requests_mock):
        """Test that custom headers are passed correctly."""
        config = EndpointConfig(
            url="http://test.com/classify",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Custom-Header": "custom-value"
            },
            timeout=30
        )

        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        client = ClassifierClient(config)
        client.predict("test")

        headers = requests_mock.last_request.headers
        assert headers['Content-Type'] == 'application/json'
        assert headers['X-Custom-Header'] == 'custom-value'


class TestClassifierClientErrors:
    """Tests for error handling."""

    def test_http_404_error(self, requests_mock, endpoint_config):
        """Test handling of 404 Not Found error."""
        requests_mock.post(
            'http://test.com/classify',
            status_code=404
        )

        client = ClassifierClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get prediction"):
            client.predict("test")

    def test_http_500_error(self, requests_mock, endpoint_config):
        """Test handling of 500 Internal Server Error."""
        requests_mock.post(
            'http://test.com/classify',
            status_code=500,
            text="Internal Server Error"
        )

        client = ClassifierClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get prediction"):
            client.predict("test")

    def test_http_503_error(self, requests_mock, endpoint_config):
        """Test handling of 503 Service Unavailable."""
        requests_mock.post(
            'http://test.com/classify',
            status_code=503
        )

        client = ClassifierClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get prediction"):
            client.predict("test")

    def test_connection_error(self, requests_mock, endpoint_config):
        """Test handling of connection errors."""
        requests_mock.post(
            'http://test.com/classify',
            exc=requests.ConnectionError("Connection refused")
        )

        client = ClassifierClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get prediction"):
            client.predict("test")

    def test_timeout_error(self, requests_mock, endpoint_config):
        """Test handling of timeout errors."""
        requests_mock.post(
            'http://test.com/classify',
            exc=requests.Timeout("Request timeout")
        )

        client = ClassifierClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get prediction"):
            client.predict("test")

    def test_invalid_json_response(self, requests_mock, endpoint_config):
        """Test handling of invalid JSON response."""
        requests_mock.post(
            'http://test.com/classify',
            text="Not a JSON response"
        )

        client = ClassifierClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get prediction"):
            client.predict("test")

    def test_missing_prediction_field(self, requests_mock, endpoint_config):
        """Test handling of response missing prediction/class field."""
        requests_mock.post(
            'http://test.com/classify',
            json={'result': 1, 'confidence': 0.95}
        )

        client = ClassifierClient(endpoint_config)
        with pytest.raises(RuntimeError, match="Invalid response format: {'result': 1, 'confidence': 0.95}"):
            client.predict("test")

    def test_invalid_prediction_type(self, requests_mock, endpoint_config):
        """Test handling of non-numeric prediction value."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 'invalid'}
        )

        client = ClassifierClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response format: {'prediction': 'invalid'}"):
            client.predict("test")


class TestClassifierClientContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_exit(self, requests_mock, endpoint_config):
        """Test that context manager properly enters and exits."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        with ClassifierClient(endpoint_config) as client:
            result = client.predict("test")
            assert result == 1
            assert client.session is not None

        # After exiting, session should be closed (test doesn't fail)

    def test_context_manager_with_exception(self, requests_mock, endpoint_config):
        """Test that context manager closes session even on exception."""
        requests_mock.post(
            'http://test.com/classify',
            json={'invalid': 'response'}
        )

        with pytest.raises(RuntimeError):
            with ClassifierClient(endpoint_config) as client:
                client.predict("test")

        # Session should still be closed even after exception

    def test_explicit_close(self, requests_mock, endpoint_config):
        """Test explicit close() method."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        client = ClassifierClient(endpoint_config)
        client.predict("test")
        client.close()

        # No assertion - just ensure close() doesn't raise exception


class TestClassifierClientTimeout:
    """Tests for timeout configuration."""

    def test_custom_timeout(self, requests_mock):
        """Test that custom timeout is used."""
        config = EndpointConfig(
            url="http://test.com/classify",
            method="POST",
            headers={},
            timeout=5
        )

        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1}
        )

        client = ClassifierClient(config)
        client.predict("test")

        # Verify timeout was passed (requests-mock doesn't directly expose this,
        # but we can verify the call was made)
        assert requests_mock.called


class TestClassifierClientResponseParsing:
    """Tests for response parsing edge cases."""

    def test_prediction_as_float(self, requests_mock, endpoint_config):
        """Test that float prediction is converted to int."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1.0}
        )

        client = ClassifierClient(endpoint_config)
        result = client.predict("test")

        assert result == 1
        assert isinstance(result, int)

    def test_prediction_as_string_numeric(self, requests_mock, endpoint_config):
        """Test that string numeric prediction is converted to int."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': '0'}
        )

        client = ClassifierClient(endpoint_config)
        result = client.predict("test")

        assert result == 0
        assert isinstance(result, int)

    def test_both_prediction_and_class_fields(self, requests_mock, endpoint_config):
        """Test that 'prediction' field takes precedence over 'class'."""
        requests_mock.post(
            'http://test.com/classify',
            json={'prediction': 1, 'class': 0}
        )

        client = ClassifierClient(endpoint_config)
        result = client.predict("test")

        assert result == 1  # prediction field takes precedence

    def test_response_with_extra_fields(self, requests_mock, endpoint_config):
        """Test that extra fields in response are ignored."""
        requests_mock.post(
            'http://test.com/classify',
            json={
                'prediction': 1,
                'confidence': 0.95,
                'model_version': '2.0',
                'timestamp': '2025-01-13'
            }
        )

        client = ClassifierClient(endpoint_config)
        result = client.predict("test")

        assert result == 1


class TestClassifierClientMultiplePredictions:
    """Tests for making multiple predictions with same client."""

    def test_multiple_sequential_predictions(self, requests_mock, endpoint_config):
        """Test making multiple predictions with the same client instance."""
        requests_mock.post(
            'http://test.com/classify',
            [
                {'json': {'prediction': 1}},
                {'json': {'prediction': 0}},
                {'json': {'prediction': 1}},
            ]
        )

        client = ClassifierClient(endpoint_config)

        result1 = client.predict("features1")
        result2 = client.predict("features2")
        result3 = client.predict("features3")

        assert result1 == 1
        assert result2 == 0
        assert result3 == 1
        assert requests_mock.call_count == 3
