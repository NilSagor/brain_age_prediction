# Brain Age Project Makefile
# Python: 3.12 | GPU: RTX 4060 Ti | CUDA: 12.4

.PHONY: help install install-dev update compile check verify clean test lint format gpu-info

# Configuration
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

# CUDA 12.4 for RTX 4060 Ti
PYTORCH_INDEX := https://download.pytorch.org/whl/cu124
TORCH_VERSION := 2.5.1

# Colors
GREEN := \033[32m
BLUE := \033[36m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

help:
	@echo "$(BLUE)🧠 Brain Age Project$(NC)"
	@echo "========================="
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install        - Create venv + install all deps"
	@echo "  make install-dev    - Also install dev tools"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Check code style"
	@echo "  make format         - Auto-format"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make verify         - Check GPU and PyTorch"
	@echo "  make gpu-info       - Show GPU details"
	@echo "  make clean          - Remove venv and caches"

# ============ VIRTUAL ENVIRONMENT ============

venv:
	@echo "$(BLUE)📦 Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	@echo "$(GREEN)✓ Virtual environment ready$(NC)"

# ============ INSTALLATION ============
# NOTE: We install in stages to avoid hash conflicts
# Stage 1: PyTorch (custom index)
# Stage 2: requirements.txt (with hashes)
# Stage 3: requirements-dev.txt (no hashes, separate to avoid conflict)

install: venv
	@echo "$(BLUE)🚀 Installing dependencies...$(NC)"
	$(PIP) install -r requirements/requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"
	@make verify


install-dev: install
	@echo "$(BLUE)🔧 Installing dev tools...$(NC)"
	$(PIP) install -r requirements/requirements-dev.txt
	@echo "$(GREEN)✓ Dev tools ready$(NC)"



# ============ DEVELOPMENT ============

# train: venv
# 	@echo "$(BLUE)🏃 Starting training...$(NC)"
# 	$(PYTHON_VENV) src/train.py --config-name baseline

test: venv
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	$(PYTEST) tests/ -v --tb=short

lint: venv
	@echo "$(BLUE)🔍 Linting...$(NC)"
	$(VENV)/bin/ruff check src/ tests/
	$(VENV)/bin/black --check src/ tests/

format: venv
	@echo "$(BLUE)✨ Formatting...$(NC)"
	$(VENV)/bin/black src/ tests/
	$(VENV)/bin/ruff check --fix src/ tests/

# ============ VERIFICATION ============

verify: venv
	@echo "$(BLUE)🔍 Verifying installation...$(NC)"
	@$(PYTHON_VENV) -c "import torch; \
		print(f'PyTorch: {torch.__version__}'); \
		print(f'CUDA: {torch.version.cuda}'); \
		print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); \
		assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null \
		&& echo "$(GREEN)✓ GPU ready$(NC)" \
		|| echo "$(RED)✗ Check failed$(NC)"

gpu-info:
	@nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,compute_cap \
		--format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

# ============ CLEANUP ============

clean:
	@echo "$(YELLOW)🧹 Cleaning...$(NC)"
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache *.egg-info dist build
	@echo "$(GREEN)✓ Cleaned$(NC)"
