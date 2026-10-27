.PHONY: help setup install download-datasets verify-datasets prepare-data train evaluate test lint format clean dashboard

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

help:
	@echo "DeepFake-AI Management Commands:"
	@echo "  make setup              - Set up project environment, verify PyTorch and directories"
	@echo "  make install            - Install production & dev dependencies"
	@echo "  make download-datasets  - Run deepfake dataset downloader / manifest generator"
	@echo "  make verify-datasets    - Verify video/image dataset integrity, frames, and checksums"
	@echo "  make prepare-data       - Run face extraction, frequency decomposition, and splits"
	@echo "  make train              - Train all spatial, frequency, and spatio-temporal deepfake models"
	@echo "  make evaluate           - Run full SOTA benchmark evaluation matrix (AUC, EER, ACC)"
	@echo "  make test               - Run unit and integration tests"
	@echo "  make lint               - Run code quality linter (ruff / flake8)"
	@echo "  make format             - Auto-format code using black and ruff"
	@echo "  make dashboard          - Launch the interactive deepfake forensics research dashboard"
	@echo "  make clean              - Clean build artifacts, pycache and temp files"

setup:
	$(PYTHON) scripts/setup_environment.py

install:
	$(PIP) install -r requirements-dev.txt

download-datasets:
	$(PYTHON) scripts/download_datasets.py --mode sample

verify-datasets:
	$(PYTHON) scripts/verify_datasets.py

prepare-data:
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) scripts/train_all.py --config configs/training.yaml

evaluate:
	$(PYTHON) scripts/evaluate_all.py

dashboard:
	$(PYTHON) scripts/run_dashboard.py

test:
	$(PYTHON) -m pytest tests/

lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
