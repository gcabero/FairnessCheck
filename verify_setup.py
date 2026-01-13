#!/usr/bin/env python3
"""
Simple script to verify the FairnessCheck installation is working.
Run this after installing to make sure everything is set up correctly.
"""

import sys


def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")

    try:
        from fairness_check import __version__
        print(f"✓ fairness_check package found (version {__version__})")
    except ImportError as e:
        print(f"✗ Failed to import fairness_check: {e}")
        return False

    try:
        from fairness_check.fairness_check_cmd import main
        print("✓ CLI module found")
    except ImportError as e:
        print(f"✗ Failed to import CLI: {e}")
        return False

    try:
        from fairness_check.config import load_config
        print("✓ Config module found")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False

    try:
        from fairness_check.ai_client import ClassifierClient
        print("✓ Client module found")
    except ImportError as e:
        print(f"✗ Failed to import client: {e}")
        return False

    try:
        from fairness_check.runner import run_fairness_check
        print("✓ Runner module found")
    except ImportError as e:
        print(f"✗ Failed to import runner: {e}")
        return False

    try:
        from fairness_check.metrics import (
            calculate_accuracy,
            calculate_demographic_parity_difference,
            calculate_equal_opportunity_difference,
        )
        print("✓ Metrics module found")
    except ImportError as e:
        print(f"✗ Failed to import metrics: {e}")
        return False

    return True


def check_dependencies():
    """Check that all required dependencies are installed."""
    print("\nChecking dependencies...")

    required = {
        "docopt": "docopt",
        "yaml": "pyyaml",
        "requests": "requests",
        "numpy": "numpy",
        "pandas": "pandas",
        "pydantic": "pydantic",
    }

    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not found")
            all_ok = False

    return all_ok


def check_example_files():
    """Check that example files exist."""
    print("\nChecking example files...")

    from pathlib import Path

    files = [
        "config.example.yaml",
        "data/test_dataset.example.csv",
    ]

    all_ok = True
    for file_path in files:
        if Path(file_path).exists():
            print(f"✓ {file_path} found")
        else:
            print(f"✗ {file_path} not found")
            all_ok = False

    return all_ok


def check_cli_command():
    """Check if the CLI command is available."""
    print("\nChecking CLI command...")

    import subprocess

    try:
        result = subprocess.run(
            ["fairness-check", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"✓ fairness-check command works")
            print(f"  Output: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ fairness-check command failed")
            return False
    except FileNotFoundError:
        print("✗ fairness-check command not found in PATH")
        print("  Try: uv pip install -e .")
        return False
    except Exception as e:
        print(f"✗ Error running fairness-check: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("FairnessCheck Installation Verification")
    print("=" * 60)

    checks = [
        ("Imports", check_imports),
        ("Dependencies", check_dependencies),
        ("Example Files", check_example_files),
        ("CLI Command", check_cli_command),
    ]

    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! Your installation is ready to use.")
        print("\nNext steps:")
        print("1. Copy config.example.yaml to config.yaml")
        print("2. Update config.yaml with your classifier endpoint")
        print("3. Prepare your test dataset CSV")
        print("4. Run: fairness-check generate config.yaml")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nTry running:")
        print("  uv sync --all-extras")
        print("  uv pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
