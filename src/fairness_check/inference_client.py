"""
Client for interacting with classifier endpoints.
"""

import logging
from typing import Any

import requests
from pydantic import BaseModel, Field, field_validator, ValidationError

from fairness_check.config import EndpointConfig

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Request payload for inference endpoint."""

    features: Any = Field(
        ...,
        description="Input features for inference (str, dict, list, or any JSON-serializable object)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"features": "user_id_123"},
                {"features": {"age": 25, "income": 50000}},
                {"features": [1.0, 2.0, 3.0]},
            ]
        }
    }


class InferenceResponse(BaseModel):
    """Response payload from inference endpoint."""

    inference: int = Field(..., description="Predicted class/label (integer)")

    @field_validator("inference", mode="before")
    @classmethod
    def validate_inference_is_int(cls, v: Any) -> int:
        """Ensure inference value is an integer."""
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            # Try to convert string to int, let Pydantic handle the rest
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"inference must be a valid integer string, got: {v}")
        if not isinstance(v, int):
            raise ValueError(f"inference must be an integer, got {type(v).__name__}: {v}")
        return v

    model_config = {"extra": "allow"}  # Allow extra fields in response (e.g., confidence, metadata)


class InferenceClient:
    """Client for making predictions via a client endpoint."""

    def __init__(self, config: EndpointConfig) -> None:
        """
        Initialize the classifier client.

        Parameters
        ----------
        config : EndpointConfig
            Endpoint configuration.
        """
        self.config = config
        self.session = requests.Session()

        # Set up headers
        self.session.headers.update(config.headers)
        if config.auth_token:
            self.session.headers["Authorization"] = f"Bearer {config.auth_token}"

        logger.info(f"Initialized InferenceClient for {config.method} {config.url}")

    def infer(self, api_input: Any) -> int:
        """
        Get the prediction or inference from a/an ML/AI system that's exposed via a
        Restful API.

        Parameters
        ----------
        api_input : Any
            The input features to send to the AI system.
            Ideally a json-serializable object (str, dict, list, etc.)

        Returns
        -------
        int
            The inference from the AI system. If we think of a classifier this could
             be the predicted class/label.

        Raises
        ------
        RuntimeError
            If the request fails or response is invalid.
        """
        try:
            # Create and validate request using Pydantic
            request = InferenceRequest(features=api_input)
            payload = request.model_dump()

            # Make HTTP request
            if self.config.method == "POST":
                response = self.session.post(
                    self.config.url,
                    json=payload,
                    timeout=self.config.timeout,
                )
            else:  # GET
                response = self.session.get(
                    self.config.url,
                    params=payload,
                    timeout=self.config.timeout,
                )

            response.raise_for_status()

            # Parse and validate response using Pydantic
            response_data = response.json()
            inference_response = InferenceResponse(**response_data)

            logger.info(
                f"Successfully validated response: inference={inference_response.inference}"
            )

            return inference_response.inference

        except requests.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise RuntimeError(f"Failed to get inference from endpoint: {e}")
        except ValidationError as e:
            # Pydantic validation error - provide clear message
            logger.error(f"Response validation failed: {e.errors()}")
            raise RuntimeError(f"Invalid response from endpoint: {e.errors()}")
        except ValueError as e:
            # JSON parsing or other value errors
            logger.error(f"Response parsing failed: {e}")
            raise RuntimeError(f"Failed to parse response: {e}")

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self) -> "InferenceClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
