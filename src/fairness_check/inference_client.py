"""
Client for interacting with classifier endpoints.
"""

from typing import Any

import requests

from fairness_check.config import EndpointConfig


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

    def infer(self, api_input: Any) -> int:
        """
        Get the prediction or inference from a/an ML/AI system that's exposed via a
        Restful API.

        Parameters
        ----------
        api_input : Any. Ideally a json-serializable object.

        Returns
        -------
        int
            Predicted value (usually 0 or 1). Although it can be extended to be a
            more complex output with maybe different subclasses depending on the
             use case we want to apply the fairness check to.

        Raises
        ------
        requests.RequestException
            If the request fails.
        ValueError/RuntimeError
            If the response is invalid and cannot be parsed.
        """
        payload = {"features": api_input}

        try:
            if self.config.method == "POST":
                response = self.session.post(
                    self.config.url, json=payload, timeout=self.config.timeout
                )
            else:  # GET
                response = self.session.get(
                    self.config.url, params=payload, timeout=self.config.timeout
                )

            response.raise_for_status()
            data = response.json()

            # Extract inference from response
            if "inference" in data:
                return int(data["inference"])
            elif "prediction" in data:
                return int(data["prediction"])
            elif "class" in data:
                return int(data["class"])
            else:
                raise ValueError(f"Invalid response format: {data}")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get inference from endpoint: {e}")
        except ValueError as e:
            raise RuntimeError(f"Invalid response format: {data}")

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self) -> "InferenceClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
