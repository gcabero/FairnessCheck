"""
Tests for fairness_check.metrics module.

Tests all fairness metric calculation functions with various scenarios including
edge cases, perfect fairness, and maximum unfairness.
"""

import pytest
import numpy as np
from fairness_check.metrics import (
    calculate_demographic_parity_difference,
    calculate_equal_opportunity_difference,
    calculate_accuracy,
)


class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""

    def test_perfect_accuracy(self):
        """Test accuracy when all predictions match true labels."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        accuracy = calculate_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(1.0), "Perfect predictions should have accuracy 1.0"

    def test_zero_accuracy(self):
        """Test accuracy when all predictions are wrong."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        accuracy = calculate_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(
            0.0
        ), "Completely wrong predictions should have accuracy 0.0"

    def test_fifty_percent_accuracy(self):
        """Test accuracy with 50% correct predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        accuracy = calculate_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(0.5), "Half correct should have accuracy 0.5"

    def test_accuracy_with_known_values(self, sample_y_true, sample_y_pred):
        """Test accuracy calculation with known values."""
        # sample_y_true = [1, 0, 1, 1, 1, 0, 0, 1]
        # sample_y_pred = [1, 0, 1, 0, 1, 1, 0, 0]
        # Correct: indices 0, 1, 2, 4, 6 (5 out of 8)
        expected_accuracy = 5 / 8
        accuracy = calculate_accuracy(sample_y_true, sample_y_pred)
        assert accuracy == pytest.approx(expected_accuracy)

    def test_accuracy_single_sample(self):
        """Test accuracy with single sample."""
        y_true = np.array([1])
        y_pred = np.array([1])
        accuracy = calculate_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(1.0)

    def test_accuracy_different_dtypes(self):
        """Test accuracy with different numpy dtypes."""
        y_true = np.array([1, 0, 1, 0], dtype=np.int32)
        y_pred = np.array([1, 0, 1, 0], dtype=np.int64)
        accuracy = calculate_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(1.0)


class TestCalculateDemographicParityDifference:
    """Tests for calculate_demographic_parity_difference function."""

    def test_perfect_fairness(self, perfect_fairness_data):
        """Test demographic parity with perfect fairness."""
        # Both groups have 2/3 positive predictions (0.6667)
        dp_diff = calculate_demographic_parity_difference(
            perfect_fairness_data["y_pred"], perfect_fairness_data["sensitive"]
        )
        assert dp_diff == pytest.approx(
            0.0, abs=1e-9
        ), "Perfect fairness should have DP difference of 0"

    def test_maximum_unfairness(self, biased_data):
        """Test demographic parity with maximum bias."""
        # Group A: 3/3 = 1.0, Group B: 0/3 = 0.0
        # Difference: 1.0 - 0.0 = 1.0
        dp_diff = calculate_demographic_parity_difference(
            biased_data["y_pred"], biased_data["sensitive"]
        )
        assert dp_diff == pytest.approx(1.0), "Maximum bias should have DP difference of 1.0"

    def test_known_difference(self):
        """Test demographic parity with known expected value."""
        # Group A: 2/4 = 0.5, Group B: 1/4 = 0.25
        # Difference: 0.5 - 0.25 = 0.25
        y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        dp_diff = calculate_demographic_parity_difference(y_pred, sensitive)
        assert dp_diff == pytest.approx(0.25)

    def test_single_group(self, edge_case_single_group):
        """Test demographic parity with only one group."""
        # Should return 0.0 (max - min = same value)
        dp_diff = calculate_demographic_parity_difference(
            edge_case_single_group["y_pred"], edge_case_single_group["sensitive"]
        )
        assert dp_diff == pytest.approx(0.0)

    def test_empty_arrays(self, edge_case_empty):
        """Test demographic parity with empty arrays."""
        dp_diff = calculate_demographic_parity_difference(
            edge_case_empty["y_pred"], edge_case_empty["sensitive"]
        )
        assert dp_diff == pytest.approx(0.0)

    def test_single_sample(self, edge_case_single_sample):
        """Test demographic parity with single sample."""
        dp_diff = calculate_demographic_parity_difference(
            edge_case_single_sample["y_pred"], edge_case_single_sample["sensitive"]
        )
        assert dp_diff == pytest.approx(0.0)

    def test_multiple_groups(self, multiple_groups_data):
        """Test demographic parity with 5 different groups."""
        # Each group has 2 samples: [1,0] = 0.5 selection rate for all
        dp_diff = calculate_demographic_parity_difference(
            multiple_groups_data["y_pred"], multiple_groups_data["sensitive"]
        )
        assert dp_diff == pytest.approx(0.0, abs=1e-9)

    def test_all_predictions_zero(self):
        """Test demographic parity when all predictions are 0."""
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])
        dp_diff = calculate_demographic_parity_difference(y_pred, sensitive)
        assert dp_diff == pytest.approx(0.0)

    def test_all_predictions_one(self):
        """Test demographic parity when all predictions are 1."""
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])
        dp_diff = calculate_demographic_parity_difference(y_pred, sensitive)
        assert dp_diff == pytest.approx(0.0)

    def test_three_groups_varying_rates(self):
        """Test demographic parity with 3 groups at different selection rates."""
        # Group A: 3/3 = 1.0, Group B: 1/3 = 0.333, Group C: 0/3 = 0.0
        # Max difference: 1.0 - 0.0 = 1.0
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
        dp_diff = calculate_demographic_parity_difference(y_pred, sensitive)
        assert dp_diff == pytest.approx(1.0)

    def test_numeric_sensitive_features(self):
        """Test demographic parity with numeric sensitive features."""
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array([1, 1, 2, 2])  # Numeric groups
        dp_diff = calculate_demographic_parity_difference(y_pred, sensitive)
        assert dp_diff == pytest.approx(0.0)


class TestCalculateEqualOpportunityDifference:
    """Tests for calculate_equal_opportunity_difference function."""

    def test_perfect_fairness(self):
        """Test equal opportunity with perfect TPR across groups."""
        # Group A: TP=2, P=2, TPR=1.0
        # Group B: TP=2, P=2, TPR=1.0
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0, abs=1e-9)

    def test_maximum_unfairness(self):
        """Test equal opportunity with maximum TPR difference."""
        # Group A: TP=2, P=2, TPR=1.0
        # Group B: TP=0, P=2, TPR=0.0
        # Difference: 1.0 - 0.0 = 1.0
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(1.0)

    def test_known_tpr_difference(self):
        """Test equal opportunity with known TPR values."""
        # Group A: TP=2, P=3, TPR=0.6667
        # Group B: TP=1, P=2, TPR=0.5
        # Difference: 0.6667 - 0.5 = 0.1667
        y_true = np.array([1, 1, 1, 0, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.1667, abs=1e-4)

    def test_no_positive_labels(self):
        """Test equal opportunity when one group has no positive labels."""
        # Group A: TP=1, P=2, TPR=0.5
        # Group B: P=0 (skip this group)
        # Should only use Group A, so difference = 0.0
        y_true = np.array([1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0)

    def test_all_groups_no_positives(self):
        """Test equal opportunity when no group has positive labels."""
        y_true = np.array([0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0)

    def test_single_group(self):
        """Test equal opportunity with single group."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0)

    def test_empty_arrays(self):
        """Test equal opportunity with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        sensitive = np.array([])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0)

    def test_multiple_groups(self):
        """Test equal opportunity with multiple groups."""
        # 5 groups, each with TPR=0.5 (1 out of 2 positives correct)
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        sensitive = np.array(
            [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "C",
                "D",
                "D",
                "D",
                "D",
                "E",
                "E",
                "E",
                "E",
            ]
        )
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0, abs=1e-9)

    def test_all_predictions_zero(self):
        """Test equal opportunity when all predictions are 0."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0)

    def test_all_predictions_one(self):
        """Test equal opportunity when all predictions are 1."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert eo_diff == pytest.approx(0.0)

    def test_different_dtypes(self):
        """Test equal opportunity with different numpy dtypes."""
        y_true = np.array([1, 1, 0, 0], dtype=np.int32)
        y_pred = np.array([1, 0, 0, 0], dtype=np.int64)
        sensitive = np.array(["A", "A", "B", "B"], dtype=object)
        eo_diff = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        # Group A: TP=1, P=2, TPR=0.5; Group B: no positives
        assert eo_diff == pytest.approx(0.0)

    def test_fixture_biased_data(self, biased_data):
        """Test equal opportunity with biased fixture data."""
        eo_diff = calculate_equal_opportunity_difference(
            biased_data["y_true"], biased_data["y_pred"], biased_data["sensitive"]
        )
        # Group A: TP=2, P=2, TPR=1.0
        # Group B: TP=0, P=1, TPR=0.0
        # Difference: 1.0
        assert eo_diff == pytest.approx(1.0)
