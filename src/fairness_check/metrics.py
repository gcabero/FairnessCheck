"""
Utility functions for fairness calculations.
"""

import numpy as np


def calculate_demographic_parity_difference(
    y_pred: np.ndarray, sensitive_features: np.ndarray
) -> float:
    """
    Calculate demographic parity difference.

    Parameters
    ----------
    y_pred : array-like
        Predicted labels (0 or 1).
    sensitive_features : array-like
        Sensitive attributes defining groups.

    Returns
    -------
    float
        Demographic parity difference (0 = perfect fairness).
    """
    groups = np.unique(sensitive_features)
    selection_rates = []

    for group in groups:
        mask = sensitive_features == group
        group_predictions = y_pred[mask]
        if len(group_predictions) > 0:
            selection_rate = np.mean(group_predictions)
            selection_rates.append(selection_rate)

    if len(selection_rates) == 0:
        return 0.0

    return float(max(selection_rates) - min(selection_rates))


def calculate_equal_opportunity_difference(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray
) -> float:
    """
    Calculate equal opportunity difference (TPR difference).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    sensitive_features : array-like
        Sensitive attributes.

    Returns
    -------
    float
        Equal opportunity difference.
    """
    groups = np.unique(sensitive_features)
    tpr_values = []

    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        # Calculate TPR (True Positive Rate)
        positives = y_true_group == 1
        if np.sum(positives) > 0:
            true_positives = np.sum((y_true_group == 1) & (y_pred_group == 1))
            tpr = true_positives / np.sum(positives)
            tpr_values.append(tpr)

    if len(tpr_values) == 0:
        return 0.0

    return float(max(tpr_values) - min(tpr_values))


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate overall accuracy.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Accuracy (0 to 1).
    """
    return float(np.mean(y_true == y_pred))
