"""
Client for interacting with classifier endpoints.
"""

from typing import Any

import requests

from fairness_check.config import EndpointConfig


class ClassifierClient:
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

    def predict(self, features: Any) -> int:
        """
        Get prediction for a single sample.

        Parameters
        ----------
        features : Any
            Features to classify (format depends on endpoint).

        Returns
        -------
        int
            Predicted class (usually 0 or 1).

        Raises
        ------
        requests.RequestException
            If the request fails.
        ValueError
            If the response is invalid.
        """
        payload = {"features": features}

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

            # Extract prediction from response
            if "prediction" in data:
                return int(data["prediction"])
            elif "class" in data:
                return int(data["class"])
            else:
                raise ValueError(f"Invalid response format: {data}")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get prediction from endpoint: {e}")
        except ValueError as e:
            raise RuntimeError(f"Invalid response format: {data}")

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self) -> "ClassifierClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
