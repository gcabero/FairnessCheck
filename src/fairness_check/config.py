"""
Configuration management for Fairness Check.

Handles loading and validating configuration files.
"""

from pathlib import Path
from typing import Any

import yaml
from yaml.parser import ParserError
from pydantic import BaseModel, Field, field_validator


class EndpointConfig(BaseModel):
    """Configuration for the classifier endpoint."""

    url: str = Field(..., description="URL of the classifier endpoint")
    method: str = Field(default="POST", description="HTTP method (GET or POST)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers to send")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    auth_token: str | None = Field(default=None, description="Optional authentication token")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate HTTP method."""
        method = v.upper()
        if method not in ["GET", "POST"]:
            raise ValueError("Method must be GET or POST")
        return method


class DatasetConfig(BaseModel):
    """Configuration for the test dataset."""

    path: str = Field(..., description="Path to the test dataset file (CSV)")
    features_column: str = Field(default="features", description="Column name containing features")
    labels_column: str = Field(default="label", description="Column name containing true labels")
    sensitive_column: str = Field(
        default="sensitive_attribute", description="Column name containing sensitive attributes"
    )


class FairnessConfig(BaseModel):
    """Configuration for fairness thresholds."""

    demographic_parity_threshold: float = Field(
        default=0.1, description="Maximum acceptable demographic parity difference"
    )
    equal_opportunity_threshold: float = Field(
        default=0.1, description="Maximum acceptable equal opportunity difference"
    )


class Config(BaseModel):
    """Main configuration for Fairness Check."""

    endpoint: EndpointConfig
    dataset: DatasetConfig
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)


def load_config(config_path: str | Path) -> Config:
    """
    Load and validate configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the configuration file.

    Returns
    -------
    Config
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.
    ValueError
        If the configuration is invalid.

    Examples
    --------
    >>> config = load_config("config.yaml")
    >>> print(config.endpoint.url)
    https://api.example.com/classify
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data: dict[str, Any] = yaml.safe_load(f)

    try:
        return Config(**config_data)
    except Exception as e:
        raise ParserError(f"Invalid configuration: {e}")
