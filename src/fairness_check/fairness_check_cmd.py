"""Fairness Check - Command line utility to implement a simple fairness evaluation tool.

Usage:
  fairness-check report <config_file> [--verbose]
  fairness-check validate <config_file>
  fairness-check --version
  fairness-check --help

Commands:
  report    Generate fairness report against the configured classifier endpoint
  validate    Checks the configuration file without running tests

Options:
  -h --help     Show this screen.
  --version     Show version.
  --verbose     Show detailed output.

Examples:
  fairness-check report config.yaml
  fairness-check report config.yaml --verbose
  fairness-check validate config.yaml
"""
import logging
import sys
from docopt import docopt

from . import __version__
from .config import load_config, Config
from .runner import run_fairness_check

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging to output to stdout.

    Parameters
    ----------
    verbose : bool
        If True, set logging level to INFO. Otherwise, WARNING.
    """
    level = logging.INFO if verbose else logging.WARNING

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main() -> None:
    """Main CLI entry point."""
    args = docopt(__doc__, version=f"Fairness Check {__version__}")

    config_file = args["<config_file>"]
    verbose = args["--verbose"]

    # Setup logging
    setup_logging(verbose)

    try:
        if args["validate"]:
            # Validate config file
            config: Config = load_config(config_file)
            print(f"✓ Configuration file '{config_file}' is valid")
            print(f"  Endpoint: {config.endpoint.url}")
            print(f"  Test dataset: {config.dataset.path}")

        elif args["report"]:
            # Generate fairness report
            config: Config = load_config(config_file)
            logger.info(f"Generating fairness report with config: {config_file}")

            results = run_fairness_check(config, verbose=verbose)

            # Display results
            print("\n" + "=" * 60)
            print("FAIRNESS TEST RESULTS")
            print("=" * 60)
            print(f"\nTotal predictions: {results['total_predictions']}")
            print(f"Accuracy: {results['accuracy']:.2%}")
            print("\nFairness Metrics:")
            for metric_name, value in results["fairness_metrics"].items():
                print(f"  {metric_name}: {value:.4f}")

            # Check thresholds
            demographic_parity_threshold = config.fairness.demographic_parity_threshold
            if results["fairness_metrics"]["demographic_parity_difference"] > demographic_parity_threshold:
                print(f"\n⚠️  Warning: Demographic parity difference exceeds {demographic_parity_threshold} threshold")
            else:
                print("\n✓ Demographic parity difference thresholds met")

            equal_odds_threshold = config.fairness.equal_opportunity_threshold
            if results["fairness_metrics"]["demographic_parity_difference"] > equal_odds_threshold:
                print(f"\n⚠️  Warning: Equal opportunity difference exceeds {demographic_parity_threshold} threshold")
            else:
                print("\n✓ Equal opportunity difference thresholds met")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if verbose:
            raise
        exit(1)


if __name__ == "__main__":
    main()
