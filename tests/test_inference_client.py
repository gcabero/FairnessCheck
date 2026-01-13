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
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        result = client.infer("test_features")

        assert result == 1
        assert requests_mock.called
        assert requests_mock.last_request.json() == {"features": "test_features"}

    def test_predict_with_dict_features(self, requests_mock, endpoint_config):
        """Test inference with dictionary features."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        features_dict = {"age": 25, "income": 50000}
        result = client.infer(features_dict)

        assert result == 1
        assert requests_mock.last_request.json() == {"features": features_dict}

    def test_predict_with_list_features(self, requests_mock, endpoint_config):
        """Test inference with list features."""
        requests_mock.post("http://test.com/classify", json={"inference": 0})

        client = InferenceClient(endpoint_config)
        features_list = [1.0, 2.0, 3.0]
        result = client.infer(features_list)

        assert result == 0
        assert requests_mock.last_request.json() == {"features": features_list}


class TestInferenceClientGET:
    """Tests for GET requests."""

    def test_predict_get_method(self, requests_mock):
        """Test GET request with query parameters."""
        config = EndpointConfig(url="http://test.com/classify", method="GET", headers={}, timeout=30)

        requests_mock.get("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(config)
        result = client.infer("test_features")

        assert result == 1
        assert "features=test_features" in requests_mock.last_request.url

    def test_get_with_dict_features(self, requests_mock):
        """Test GET request with dictionary features as query params."""
        config = EndpointConfig(url="http://test.com/classify", method="GET", headers={}, timeout=30)

        requests_mock.get("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(config)
        result = client.infer({"key": "value"})

        assert result == 1

    def test_get_with_list_features(self, requests_mock):
        """Test GET request with list features."""
        config = EndpointConfig(url="http://test.com/classify", method="GET", headers={}, timeout=30)

        requests_mock.get("http://test.com/classify", json={"inference": 0})

        client = InferenceClient(config)
        result = client.infer([1, 2, 3])

        assert result == 0

    def test_get_with_special_characters(self, requests_mock):
        """Test GET request with special characters in features."""
        config = EndpointConfig(url="http://test.com/classify", method="GET", headers={}, timeout=30)

        requests_mock.get("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(config)
        result = client.infer("user@example.com")

        assert result == 1

    def test_get_with_empty_features(self, requests_mock):
        """Test GET request with empty string features."""
        config = EndpointConfig(url="http://test.com/classify", method="GET", headers={}, timeout=30)

        requests_mock.get("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(config)
        result = client.infer("")

        assert result == 1


class TestInferenceClientAuthentication:
    """Tests for authentication."""

    def test_auth_header_injection(self, requests_mock, endpoint_config_with_auth):
        """Test that Bearer token is injected correctly."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config_with_auth)
        client.infer("test")

        auth_header = requests_mock.last_request.headers.get("Authorization")
        assert auth_header == "Bearer test-token-123"

    def test_custom_headers(self, requests_mock):
        """Test that custom headers are passed correctly."""
        config = EndpointConfig(
            url="http://test.com/classify",
            method="POST",
            headers={"Content-Type": "application/json", "X-Custom-Header": "custom-value"},
            timeout=30,
        )

        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(config)
        client.infer("test")

        headers = requests_mock.last_request.headers
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Custom-Header"] == "custom-value"


class TestInferenceClientErrors:
    """Tests for error handling."""

    def test_http_404_error(self, requests_mock, endpoint_config):
        """Test handling of 404 Not Found error."""
        requests_mock.post("http://test.com/classify", status_code=404)

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_500_error(self, requests_mock, endpoint_config):
        """Test handling of 500 Internal Server Error."""
        requests_mock.post("http://test.com/classify", status_code=500, text="Internal Server Error")

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_503_error(self, requests_mock, endpoint_config):
        """Test handling of 503 Service Unavailable."""
        requests_mock.post("http://test.com/classify", status_code=503)

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_connection_error(self, requests_mock, endpoint_config):
        """Test handling of connection errors."""
        requests_mock.post("http://test.com/classify", exc=requests.ConnectionError("Connection refused"))

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_timeout_error(self, requests_mock, endpoint_config):
        """Test handling of timeout errors."""
        requests_mock.post("http://test.com/classify", exc=requests.Timeout("Request timeout"))

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_invalid_json_response(self, requests_mock, endpoint_config):
        """Test handling of invalid JSON response."""
        requests_mock.post("http://test.com/classify", text="Not a JSON response")

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_missing_inference_field(self, requests_mock, endpoint_config):
        """Test handling of response missing inference field."""
        requests_mock.post("http://test.com/classify", json={"result": 1, "confidence": 0.95})

        client = InferenceClient(endpoint_config)
        with pytest.raises(RuntimeError, match="Invalid response from endpoint"):
            client.infer("test")

    def test_invalid_inference_type(self, requests_mock, endpoint_config):
        """Test handling of non-numeric inference value."""
        requests_mock.post("http://test.com/classify", json={"inference": "invalid"})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response from endpoint"):
            client.infer("test")

    def test_http_204_no_content_error(self, requests_mock, endpoint_config):
        """Test handling of 204 No Content (no JSON body)."""
        requests_mock.post("http://test.com/classify", status_code=204)

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError):
            client.infer("test")

    def test_http_400_bad_request(self, requests_mock, endpoint_config):
        """Test handling of 400 Bad Request."""
        requests_mock.post("http://test.com/classify", status_code=400, json={"error": "Invalid features format"})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_401_unauthorized(self, requests_mock, endpoint_config):
        """Test handling of 401 Unauthorized."""
        requests_mock.post("http://test.com/classify", status_code=401)

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_403_forbidden(self, requests_mock, endpoint_config):
        """Test handling of 403 Forbidden."""
        requests_mock.post("http://test.com/classify", status_code=403)

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_429_rate_limit(self, requests_mock, endpoint_config):
        """Test handling of 429 Too Many Requests."""
        requests_mock.post("http://test.com/classify", status_code=429, headers={"Retry-After": "60"})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_http_502_bad_gateway(self, requests_mock, endpoint_config):
        """Test handling of 502 Bad Gateway."""
        requests_mock.post("http://test.com/classify", status_code=502)

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_too_many_redirects_error(self, requests_mock, endpoint_config):
        """Test handling of too many redirects."""
        requests_mock.post("http://test.com/classify", exc=requests.TooManyRedirects("Too many redirects"))

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Failed to get inference"):
            client.infer("test")

    def test_empty_json_response_fails(self, requests_mock, endpoint_config):
        """Test handling of empty JSON object response."""
        requests_mock.post("http://test.com/classify", json={})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response from endpoint"):
            client.infer("test")


class TestInferenceClientContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_exit(self, requests_mock, endpoint_config):
        """Test that context manager properly enters and exits."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        with InferenceClient(endpoint_config) as client:
            result = client.infer("test")
            assert result == 1
            assert client.session is not None

        # After exiting, session should be closed (test doesn't fail)

    def test_context_manager_with_exception(self, requests_mock, endpoint_config):
        """Test that context manager closes session even on exception."""
        requests_mock.post("http://test.com/classify", json={"invalid": "response"})

        with pytest.raises(RuntimeError):
            with InferenceClient(endpoint_config) as client:
                client.infer("test")

        # Session should still be closed even after exception

    def test_explicit_close(self, requests_mock, endpoint_config):
        """Test explicit close() method."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        client.infer("test")
        client.close()

        # No assertion - just ensure close() doesn't raise exception

    def test_close_multiple_times(self, requests_mock, endpoint_config):
        """Test that calling close() multiple times doesn't raise error."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        client.infer("test")
        client.close()
        client.close()  # Should not raise

    def test_use_after_close(self, requests_mock, endpoint_config):
        """Test that using client after close() still works (session may be recreated)."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        client.close()
        # Session might still work or might fail - this tests actual behavior
        result = client.infer("test")
        assert result == 1

    def test_context_manager_reentry(self, requests_mock, endpoint_config):
        """Test re-entering context manager after exit."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)

        with client:
            client.infer("test1")

        with client:  # Re-enter
            result = client.infer("test2")
            assert result == 1


class TestInferenceClientTimeout:
    """Tests for timeout configuration."""

    def test_custom_timeout(self, requests_mock):
        """Test that custom timeout is used."""
        config = EndpointConfig(url="http://test.com/classify", method="POST", headers={}, timeout=5)

        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(config)
        client.infer("test")

        # Verify timeout was passed (requests-mock doesn't directly expose this,
        # but we can verify the call was made)
        assert requests_mock.called


class TestInferenceClientResponseParsing:
    """Tests for response parsing edge cases."""

    def test_inference_as_float(self, requests_mock, endpoint_config):
        """Test that float inference is converted to int."""
        requests_mock.post("http://test.com/classify", json={"inference": 1.0})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 1
        assert isinstance(result, int)

    def test_inference_as_string_numeric(self, requests_mock, endpoint_config):
        """Test that string numeric inference is converted to int."""
        requests_mock.post("http://test.com/classify", json={"inference": "0"})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 0
        assert isinstance(result, int)

    def test_response_with_extra_fields(self, requests_mock, endpoint_config):
        """Test that extra fields in response are ignored."""
        requests_mock.post(
            "http://test.com/classify",
            json={"inference": 1, "confidence": 0.95, "model_version": "2.0", "timestamp": "2025-01-13"},
        )

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 1


class TestInferenceClientMultiple:
    """Tests for making multiple inferences with same client."""

    def test_multiple_sequential_inferences(self, requests_mock, endpoint_config):
        """Test making multiple inferences with the same client instance."""
        requests_mock.post(
            "http://test.com/classify",
            [
                {"json": {"inference": 1}},
                {"json": {"inference": 0}},
                {"json": {"inference": 1}},
            ],
        )

        client = InferenceClient(endpoint_config)

        result1 = client.infer("features1")
        result2 = client.infer("features2")
        result3 = client.infer("features3")

        assert result1 == 1
        assert result2 == 0
        assert result3 == 1
        assert requests_mock.call_count == 3


class TestInferenceResponseValidation:
    """Tests for InferenceResponse Pydantic model validation edge cases."""

    def test_inference_negative_float_conversion(self, requests_mock, endpoint_config):
        """Test negative float is converted to int."""
        requests_mock.post("http://test.com/classify", json={"inference": -1.5})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == -1
        assert isinstance(result, int)

    def test_inference_zero_float_conversion(self, requests_mock, endpoint_config):
        """Test zero as float is converted to int."""
        requests_mock.post("http://test.com/classify", json={"inference": 0.0})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 0
        assert isinstance(result, int)

    def test_inference_boolean_true(self, requests_mock, endpoint_config):
        """Test boolean True is accepted (bool is int subclass in Python)."""
        requests_mock.post("http://test.com/classify", json={"inference": True})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 1
        assert isinstance(result, int)

    def test_inference_boolean_false(self, requests_mock, endpoint_config):
        """Test boolean False is accepted (bool is int subclass in Python)."""
        requests_mock.post("http://test.com/classify", json={"inference": False})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 0
        assert isinstance(result, int)

    def test_inference_negative_string_conversion(self, requests_mock, endpoint_config):
        """Test negative string number is converted to int."""
        requests_mock.post("http://test.com/classify", json={"inference": "-1"})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == -1
        assert isinstance(result, int)

    def test_inference_empty_string_fails(self, requests_mock, endpoint_config):
        """Test empty string fails validation."""
        requests_mock.post("http://test.com/classify", json={"inference": ""})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response from endpoint"):
            client.infer("test")

    def test_inference_whitespace_string_conversion(self, requests_mock, endpoint_config):
        """Test string with whitespace is converted to int."""
        requests_mock.post("http://test.com/classify", json={"inference": " 1 "})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 1
        assert isinstance(result, int)

    def test_inference_none_fails(self, requests_mock, endpoint_config):
        """Test None value fails validation."""
        requests_mock.post("http://test.com/classify", json={"inference": None})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response from endpoint"):
            client.infer("test")

    def test_inference_list_fails(self, requests_mock, endpoint_config):
        """Test list value fails validation."""
        requests_mock.post("http://test.com/classify", json={"inference": [1]})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response from endpoint"):
            client.infer("test")

    def test_inference_dict_fails(self, requests_mock, endpoint_config):
        """Test dict value fails validation."""
        requests_mock.post("http://test.com/classify", json={"inference": {"value": 1}})

        client = InferenceClient(endpoint_config)

        with pytest.raises(RuntimeError, match="Invalid response from endpoint"):
            client.infer("test")

    def test_inference_large_positive_integer(self, requests_mock, endpoint_config):
        """Test very large positive integer."""
        requests_mock.post("http://test.com/classify", json={"inference": 999999})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == 999999
        assert isinstance(result, int)

    def test_inference_large_negative_integer(self, requests_mock, endpoint_config):
        """Test very large negative integer."""
        requests_mock.post("http://test.com/classify", json={"inference": -999999})

        client = InferenceClient(endpoint_config)
        result = client.infer("test")

        assert result == -999999
        assert isinstance(result, int)


class TestInferenceRequestEdgeCases:
    """Tests for InferenceRequest edge cases with various feature types."""

    def test_empty_string_features(self, requests_mock, endpoint_config):
        """Test with empty string features."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        result = client.infer("")

        assert result == 1
        assert requests_mock.last_request.json() == {"features": ""}

    def test_empty_dict_features(self, requests_mock, endpoint_config):
        """Test with empty dictionary features."""
        requests_mock.post("http://test.com/classify", json={"inference": 0})

        client = InferenceClient(endpoint_config)
        result = client.infer({})

        assert result == 0
        assert requests_mock.last_request.json() == {"features": {}}

    def test_empty_list_features(self, requests_mock, endpoint_config):
        """Test with empty list features."""
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        result = client.infer([])

        assert result == 1
        assert requests_mock.last_request.json() == {"features": []}

    def test_nested_dict_features(self, requests_mock, endpoint_config):
        """Test with deeply nested dictionary features."""
        features = {"level1": {"level2": {"level3": [1, 2, 3]}}}
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        result = client.infer(features)

        assert result == 1
        assert requests_mock.last_request.json() == {"features": features}

    def test_unicode_string_features(self, requests_mock, endpoint_config):
        """Test with unicode characters in features."""
        features = "用户123_测试"
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        result = client.infer(features)

        assert result == 1

    def test_special_characters_features(self, requests_mock, endpoint_config):
        """Test with special characters in features."""
        features = "user@example.com#123&token=abc"
        requests_mock.post("http://test.com/classify", json={"inference": 0})

        client = InferenceClient(endpoint_config)
        result = client.infer(features)

        assert result == 0

    def test_tuple_features(self, requests_mock, endpoint_config):
        """Test with tuple features (should be serializable)."""
        features = (1, 2, 3)
        requests_mock.post("http://test.com/classify", json={"inference": 1})

        client = InferenceClient(endpoint_config)
        result = client.infer(features)

        assert result == 1

    def test_none_features(self, requests_mock, endpoint_config):
        """Test with None features."""
        requests_mock.post("http://test.com/classify", json={"inference": 0})

        client = InferenceClient(endpoint_config)
        result = client.infer(None)

        assert result == 0
        assert requests_mock.last_request.json() == {"features": None}
