# Contributing to DeepFake-AI

Thank you for your interest in contributing to DeepFake-AI! We welcome contributions ranging from novel detector backbones, frequency analysis techniques, and dataset adapters to documentation and unit tests.

## Development Workflow

1. **Fork & Branch**: Create a feature branch with a descriptive name (`git checkout -b feature/spatial-frequency-dual-stream`).
2. **Environment Setup**: Run `make setup` and `make install` to configure dependencies.
3. **Coding Standards**:
   - Format with `black` and lint with `ruff`.
   - Adhere to PEP 8 standards with type hints across all core methods.
   - Maintain reproducible random seeds in dataset splitters.
4. **Testing**:
   - Write unit tests under `tests/unit/` for new neural network blocks or preprocessing filters.
   - Run `make test` before submitting pull requests.
5. **Research Integrity**:
   - Ensure all benchmark claims are backed by seed-fixed evaluation runs.
   - Avoid identity leakage: training and test splits must not share actors or source identities.
