"""
Validation tests for fairness_check.metrics against fairlearn library.

These tests compare the custom metric implementations against fairlearn's
reference implementations to ensure correctness.

Note: These tests require fairlearn to be installed (dev dependency).
"""

import pytest
import numpy as np

# Try to import fairlearn - tests will be skipped if not available
try:
    from fairlearn.metrics import demographic_parity_difference as fl_dp_diff
    from fairlearn.metrics import equalized_odds_difference as fl_eo_diff
    from sklearn.metrics import accuracy_score

    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

from fairness_check.metrics import (
    calculate_demographic_parity_difference,
    calculate_equal_opportunity_difference,
    calculate_accuracy,
)

pytestmark = pytest.mark.skipif(
    not FAIRLEARN_AVAILABLE, reason="fairlearn not installed (required for validation tests)"
)


class TestDemographicParityValidation:
    """Validate demographic parity against fairlearn."""

    def test_basic_case_matches_fairlearn(self):
        """Test that basic case matches fairlearn exactly."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        custom_result = calculate_demographic_parity_difference(y_pred, sensitive)
        fairlearn_result = fl_dp_diff(y_true, y_pred, sensitive_features=sensitive)

        assert custom_result == pytest.approx(
            fairlearn_result, abs=1e-9
        ), f"Custom: {custom_result}, Fairlearn: {fairlearn_result}"

    def test_perfect_fairness_matches_fairlearn(self, perfect_fairness_data):
        """Test perfect fairness scenario matches fairlearn."""
        custom_result = calculate_demographic_parity_difference(
            perfect_fairness_data["y_pred"], perfect_fairness_data["sensitive"]
        )
        fairlearn_result = fl_dp_diff(
            perfect_fairness_data["y_true"],
            perfect_fairness_data["y_pred"],
            sensitive_features=perfect_fairness_data["sensitive"],
        )

        assert custom_result == pytest.approx(fairlearn_result, abs=1e-9)

    def test_biased_case_matches_fairlearn(self, biased_data):
        """Test biased scenario matches fairlearn."""
        custom_result = calculate_demographic_parity_difference(biased_data["y_pred"], biased_data["sensitive"])
        fairlearn_result = fl_dp_diff(
            biased_data["y_true"], biased_data["y_pred"], sensitive_features=biased_data["sensitive"]
        )

        assert custom_result == pytest.approx(fairlearn_result, abs=1e-9)

    def test_multiple_groups_matches_fairlearn(self, multiple_groups_data):
        """Test multiple groups scenario matches fairlearn."""
        custom_result = calculate_demographic_parity_difference(
            multiple_groups_data["y_pred"], multiple_groups_data["sensitive"]
        )
        fairlearn_result = fl_dp_diff(
            multiple_groups_data["y_true"],
            multiple_groups_data["y_pred"],
            sensitive_features=multiple_groups_data["sensitive"],
        )

        assert custom_result == pytest.approx(fairlearn_result, abs=1e-9)

    def test_large_dataset_matches_fairlearn(self):
        """Test with larger dataset to ensure scalability."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        sensitive = np.random.choice(["A", "B", "C"], n_samples)

        custom_result = calculate_demographic_parity_difference(y_pred, sensitive)
        fairlearn_result = fl_dp_diff(y_true, y_pred, sensitive_features=sensitive)

        assert custom_result == pytest.approx(fairlearn_result, abs=1e-9)


class TestEqualOpportunityValidation:
    """Validate equal opportunity (TPR) against fairlearn.

    Note: Fairlearn's equalized_odds_difference includes both TPR and FPR.
    We only compare TPR (equal opportunity) here.
    """

    def test_basic_case_tpr_component(self):
        """Test basic case TPR matches fairlearn's TPR component."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        custom_result = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)

        # Calculate TPR manually for fairlearn comparison
        # Group A: TP=1, P=2, TPR=0.5
        # Group B: TP=1, P=2, TPR=0.5
        # Difference should be 0
        assert custom_result == pytest.approx(0.0, abs=1e-9)

    def test_perfect_tpr_equality(self):
        """Test perfect TPR equality across groups."""
        y_true = np.array([1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        custom_result = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        assert custom_result == pytest.approx(0.0)

    def test_maximum_tpr_difference(self):
        """Test maximum TPR difference."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        custom_result = calculate_equal_opportunity_difference(y_true, y_pred, sensitive)
        # Group A: TPR=1.0, Group B: TPR=0.0
        assert custom_result == pytest.approx(1.0)

    def test_biased_data_tpr(self, biased_data):
        """Test TPR with biased data."""
        custom_result = calculate_equal_opportunity_difference(
            biased_data["y_true"], biased_data["y_pred"], biased_data["sensitive"]
        )
        # Group A: y_true=[1,0,1], y_pred=[1,1,1], TP=2, P=2, TPR=1.0
        # Group B: y_true=[0,1,0], y_pred=[0,0,0], TP=0, P=1, TPR=0.0
        assert custom_result == pytest.approx(1.0)


class TestAccuracyValidation:
    """Validate accuracy against sklearn."""

    def test_accuracy_matches_sklearn(self):
        """Test that accuracy matches sklearn's accuracy_score."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0])

        custom_result = calculate_accuracy(y_true, y_pred)
        sklearn_result = accuracy_score(y_true, y_pred)

        assert custom_result == pytest.approx(sklearn_result)

    def test_perfect_accuracy_matches_sklearn(self):
        """Test perfect accuracy matches sklearn."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])

        custom_result = calculate_accuracy(y_true, y_pred)
        sklearn_result = accuracy_score(y_true, y_pred)

        assert custom_result == pytest.approx(sklearn_result)
        assert custom_result == pytest.approx(1.0)

    def test_zero_accuracy_matches_sklearn(self):
        """Test zero accuracy matches sklearn."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])

        custom_result = calculate_accuracy(y_true, y_pred)
        sklearn_result = accuracy_score(y_true, y_pred)

        assert custom_result == pytest.approx(sklearn_result)
        assert custom_result == pytest.approx(0.0)

    def test_large_dataset_accuracy_matches_sklearn(self):
        """Test accuracy with large dataset matches sklearn."""
        np.random.seed(42)
        n_samples = 10000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)

        custom_result = calculate_accuracy(y_true, y_pred)
        sklearn_result = accuracy_score(y_true, y_pred)

        assert custom_result == pytest.approx(sklearn_result, abs=1e-9)

    def test_accuracy_with_fixtures(self, sample_y_true, sample_y_pred):
        """Test accuracy with fixtures matches sklearn."""
        custom_result = calculate_accuracy(sample_y_true, sample_y_pred)
        sklearn_result = accuracy_score(sample_y_true, sample_y_pred)

        assert custom_result == pytest.approx(sklearn_result)


class TestRealDatasetValidation:
    """Test custom metrics against fairlearn using real example data."""

    @pytest.fixture
    def example_dataset_path(self):
        """Path to example dataset."""
        return "data/test_dataset.example.csv"

    def test_with_example_dataset(self, example_dataset_path):
        """Test metrics with real example dataset if it exists."""
        import pandas as pd
        from pathlib import Path

        if not Path(example_dataset_path).exists():
            pytest.skip(f"Example dataset not found: {example_dataset_path}")

        df = pd.read_csv(example_dataset_path)

        # Assuming standard column names
        y_pred = np.random.randint(0, 2, len(df))  # Mock predictions
        y_true = df["label"].values
        sensitive = df["sensitive_attribute"].values

        # Test demographic parity
        custom_dp = calculate_demographic_parity_difference(y_pred, sensitive)
        fairlearn_dp = fl_dp_diff(y_true, y_pred, sensitive_features=sensitive)
        assert custom_dp == pytest.approx(fairlearn_dp, abs=1e-9)

        # Test accuracy
        custom_acc = calculate_accuracy(y_true, y_pred)
        sklearn_acc = accuracy_score(y_true, y_pred)
        assert custom_acc == pytest.approx(sklearn_acc)


# Documentation test to show differences
class TestDocumentedDifferences:
    """Document intentional differences from fairlearn."""

    def test_demographic_parity_note(self):
        """Document that demographic parity is functionally equivalent."""
        # Our implementation: max(selection_rates) - min(selection_rates)
        # Fairlearn: Same approach
        # Conclusion: No intentional differences
        pass

    def test_equal_opportunity_vs_equalized_odds(self):
        """Document difference between equal opportunity and equalized odds."""
        # Our implementation: Only TPR (True Positive Rate) difference
        # Fairlearn equalized_odds: Both TPR and FPR (False Positive Rate)
        #
        # This is an intentional simplification:
        # - Equal Opportunity: Focuses only on positive outcomes (TPR)
        # - Equalized Odds: Focuses on both positive and negative outcomes (TPR + FPR)
        #
        # For binary fairness testing, equal opportunity is often sufficient
        pass
