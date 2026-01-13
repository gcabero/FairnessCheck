# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Additional fairness metrics (calibration, counterfactual fairness)
- Support for multi-class classification
- Visualization tools for fairness metrics
- Integration with popular ML frameworks (scikit-learn, PyTorch, TensorFlow)

## [0.1.0] - 2025-01-12

### Added
- Initial release of Fairness Check
- Custom fairness metric implementations:
  - `demographic_parity_difference`: Calculate demographic parity difference
  - `demographic_parity_ratio`: Calculate demographic parity ratio (80% rule)
- Fairlearn wrapper functions:
  - `equalized_odds_difference`: Equalized odds metric
  - `equal_opportunity_difference`: Equal opportunity metric
  - `selection_rate`: Selection rate calculation
  - `disparate_impact_ratio`: Disparate impact ratio
- Utility functions:
  - `validate_inputs`: Input validation for metrics
  - `ensure_binary`: Binary label validation
  - `group_statistics`: Calculate statistics per group
  - `confusion_matrix_by_group`: Confusion matrices per group
- Comprehensive test suite with pytest
- Type hints throughout the codebase
- Pydantic-based input validation
- GitHub Actions CI/CD pipelines:
  - Automated testing on Python 3.10, 3.11, 3.12, 3.13
  - Code quality checks (Black)
  - Automated publishing to PyPI
- Documentation:
  - README with quick start guide
  - Detailed docstrings with mathematical formulas
  - Contributing guidelines
  - MIT License

### Technical Details
- Python 3.10+ required
- Uses UV package manager
- Src-layout project structure
- Type annotations throughout
- Code formatted with Black (line length: 100)
- Test coverage tracking with pytest-cov

[Unreleased]: https://github.com/yourusername/fairness-check/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/fairness-check/releases/tag/v0.1.0
