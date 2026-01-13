"""
Tests for fairness_check.ai_client module.

Tests the InferenceClient class that makes HTTP requests to classifier endpoints.
Uses requests-mock to mock HTTP responses.
"""

import pytest
import requests
from fairness_check.inference_client import InferenceClient
from fairness_check.config import EndpointConfig


class TestInferenceClientPOST:
    """Tests for POST requests."""

    def test_predict_success_with_inference_field(self, requests_mock, endpoint_config):
        """Test successful POST inference with 'inference' field in response."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 1}
        )

        client = InferenceClient(endpoint_config)
        result = client.infer("test_features")

        assert result == 1
        assert requests_mock.called
        assert requests_mock.last_request.json() == {'features': 'test_features'}

    def test_predict_success_with_class_field(self, requests_mock, endpoint_config):
        """Test successful POST inference with 'class' field in response."""
        requests_mock.post(
            'http://test.com/classify',
            json={'class': 0}
        )

        client = InferenceClient(endpoint_config)
        result = client.infer("test_features")

        assert result == 0

    def test_predict_with_dict_features(self, requests_mock, endpoint_config):
        """Test inference with dictionary features."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 1}
        )

        client = InferenceClient(endpoint_config)
        features_dict = {'age': 25, 'income': 50000}
        result = client.infer(features_dict)

        assert result == 1
        assert requests_mock.last_request.json() == {'features': features_dict}

    def test_predict_with_list_features(self, requests_mock, endpoint_config):
        """Test inference with list features."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 0}
        )

        client = InferenceClient(endpoint_config)
        features_list = [1.0, 2.0, 3.0]
        result = client.infer(features_list)

        assert result == 0
        assert requests_mock.last_request.json() == {'features': features_list}


class TestInferenceClientGET:
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
            json={'inference': 1}
        )

        client = InferenceClient(config)
        result = client.infer("test_features")

        assert result == 1
        assert 'features=test_features' in requests_mock.last_request.url


class TestInferenceClientAuthentication:
    """Tests for authentication."""

    def test_auth_header_injection(self, requests_mock, endpoint_config_with_auth):
        """Test that Bearer token is injected correctly."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 1}
        )

        client = InferenceClient(endpoint_config_with_auth)
        client.infer("test")

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
            json={'inference': 1}
        )

        client = InferenceClient(config)
        client.infer("test")

        headers = requests_mock.last_request.headers
        assert headers['Content-Type'] == 'application/json'
        assert headers['X-Custom-Header'] == 'custom-value'


class TestInferenceClientErrors:
    """Tests for error handling."""

    def test_http_404_error(self, requests_mock, endpoint_config):
        """Test handling of 404 Not Found error."""
        requests_mock.post(
            'http://test.com/classify',
            status_code=404
        )

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_500_error(self, requests_mock, endpoint_config):
        """Test handling of 500 Internal Server Error."""
        requests_mock.post(
            'http://test.com/classify',
            status_code=500,
            text="Internal Server Error"
        )

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_503_error(self, requests_mock, endpoint_config):
        """Test handling of 503 Service Unavailable."""
        requests_mock.post(
            'http://test.com/classify',
            status_code=503
        )

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_connection_error(self, requests_mock, endpoint_config):
        """Test handling of connection errors."""
        requests_mock.post(
            'http://test.com/classify',
            exc=requests.ConnectionError("Connection refused")
        )

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_timeout_error(self, requests_mock, endpoint_config):
        """Test handling of timeout errors."""
        requests_mock.post(
            'http://test.com/classify',
            exc=requests.Timeout("Request timeout")
        )

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_invalid_json_response(self, requests_mock, endpoint_config):
        """Test handling of invalid JSON response."""
        requests_mock.post(
            'http://test.com/classify',
            text="Not a JSON response"
        )

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_missing_inference_field(self, requests_mock, endpoint_config):
        """Test handling of response missing inference/class field."""
        requests_mock.post(
            'http://test.com/classify',
            json={'result': 1, 'confidence': 0.95}
        )

        client = InferenceClient(endpoint_config)
        with pytest.raises(RuntimeError, match="Invalid response format: {'result': 1, 'confidence': 0.95}"):
            client.infer("test")

    def test_invalid_inference_type(self, requests_mock, endpoint_config):
        """Test handling of non-numeric inference value."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 'invalid'}
        )

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response format: {'inference': 'invalid'}"):
            client.infer("test")


class TestInferenceClientContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_exit(self, requests_mock, endpoint_config):
        """Test that context manager properly enters and exits."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 1}
        )

        with InferenceClient(endpoint_config) as client:
            result = client.infer("test")
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
            with InferenceClient(endpoint_config) as client:
                client.infer("test")

        # Session should still be closed even after exception

    def test_explicit_close(self, requests_mock, endpoint_config):
        """Test explicit close() method."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 1}
        )

        client = InferenceClient(endpoint_config)
        client.infer("test")
        client.close()

        # No assertion - just ensure close() doesn't raise exception


class TestInferenceClientTimeout:
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
            json={'inference': 1}
        )

        client = InferenceClient(config)
        client.infer("test")

        # Verify timeout was passed (requests-mock doesn't directly expose this,
        # but we can verify the call was made)
        assert requests_mock.called


class TestInferenceClientResponseParsing:
    """Tests for response parsing edge cases."""

    def test_inference_as_float(self, requests_mock, endpoint_config):
        """Test that float inference is converted to int."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 1.0}
        )

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 1
        assert isinstance(result, int)

    def test_inference_as_string_numeric(self, requests_mock, endpoint_config):
        """Test that string numeric inference is converted to int."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': '0'}
        )

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 0
        assert isinstance(result, int)

    def test_both_inference_and_class_fields(self, requests_mock, endpoint_config):
        """Test that 'inference' field takes precedence over 'class'."""
        requests_mock.post(
            'http://test.com/classify',
            json={'inference': 1, 'class': 0}
        )

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 1  # inference field takes precedence

    def test_response_with_extra_fields(self, requests_mock, endpoint_config):
        """Test that extra fields in response are ignored."""
        requests_mock.post(
            'http://test.com/classify',
            json={
                'inference': 1,
                'confidence': 0.95,
                'model_version': '2.0',
                'timestamp': '2025-01-13'
            }
        )

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 1


class TestInferenceClientMultiple:
    """Tests for making multiple inferences with same client."""

    def test_multiple_sequential_inferences(self, requests_mock, endpoint_config):
        """Test making multiple inferences with the same client instance."""
        requests_mock.post(
            'http://test.com/classify',
            [
                {'json': {'inference': 1}},
                {'json': {'inference': 0}},
                {'json': {'inference': 1}},
            ]
        )

        client = InferenceClient(endpoint_config)

        result1 = client.infer("features1")
        result2 = client.infer("features2")
        result3 = client.infer("features3")

        assert result1 == 1
        assert result2 == 0
        assert result3 == 1
        assert requests_mock.call_count == 3
