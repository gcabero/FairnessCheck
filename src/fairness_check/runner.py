"""
Test runner for fairness evaluation.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np

from .inference_client import InferenceClient
from .config import Config
from .metrics import (
    calculate_accuracy,
    calculate_demographic_parity_difference,
    calculate_equal_opportunity_difference,
)

logger = logging.getLogger(__name__)


def load_dataset(config: Config) -> pd.DataFrame:
    """
    Load the test dataset from file.

    Parameters
    ----------
    config : Config
        Configuration containing dataset information.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    return pd.read_csv(config.dataset.path)


def run_fairness_check(config: Config, verbose: bool = False) -> dict[str, Any]:
    """
    Run fairness tests against the configured endpoint.

    Parameters
    ----------
    config : Config
        Configuration for the tests.
    verbose : bool
        Whether to print detailed progress.

    Returns
    -------
    dict
        Test results containing metrics and statistics.
    """
    # Load dataset
    if verbose:
        logger.info("Loading test dataset...")
    df = load_dataset(config)

    features_col = config.dataset.features_column
    labels_col = config.dataset.labels_column
    sensitive_col = config.dataset.sensitive_column

    # Validate columns exist
    for col in [features_col, labels_col, sensitive_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    # Extract data
    features_list = df[features_col].tolist()
    y_true = df[labels_col].values
    sensitive_features = df[sensitive_col].values

    # Get answers from the AI system we want to evaluate fairness accross
    y_pred = get_predictions(config, features_list)

    # Calculate fairness metrics
    results = calculate_metrics(config, sensitive_features, y_pred, y_true)

    return results


def get_predictions(config, features_list: list[Any], verbose=None) -> np.ndarray:
    """Given the inputs to the model it calls the system to evaluate via restful
    API to get the predictions"""

    if verbose:
        logger.info("Calling endpoint to get model's answers ...")
    predictions = []
    with InferenceClient(config.endpoint) as client:
        for i, features in enumerate(features_list):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(features_list)} samples")

            pred = client.infer(features)
            predictions.append(pred)
    y_pred = np.array(predictions)
    return y_pred


def calculate_metrics(config, sensitive_features, y_pred, y_true, verbose=None):
    """Given the results from the model, the labelled correct answers and the
    sensitive data that we are calculating fairness against it, it calculates some
    statistical measures relevant to fairness evaluation"""

    if verbose:
        logger.info("Calculating fairness metrics...")

    accuracy = calculate_accuracy(y_true, y_pred)
    dp_diff = calculate_demographic_parity_difference(y_pred, sensitive_features)
    eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive_features)
    results = {
        "total_predictions": len(y_pred),
        "accuracy": accuracy,
        "fairness_metrics": {
            "demographic_parity_difference": dp_diff,
            "equal_opportunity_difference": eo_diff,
        },
        "thresholds_met": {
            "demographic_parity": dp_diff <= config.fairness.demographic_parity_threshold,
            "equal_opportunity": eo_diff <= config.fairness.equal_opportunity_threshold,
        },
    }
    return results
