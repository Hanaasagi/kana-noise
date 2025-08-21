# Makefile for KANA project
# Using .DEFAULT_GOAL to set the default target to help
.DEFAULT_GOAL := help

# Define variables
PYTHON := python
UV := uv
UVX := uvx
VIDEOS_DIR := ./videos
RUNS_DIR := ./runs
SCRIPTS_DIR := ./scripts
SRC_DIR := src

# Color definitions (for output beautification)
RESET := \033[0m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
BOLD := \033[1m

.PHONY: help
help: ## Display help information
	@echo "$(BOLD)Commands$(RESET)"
	@echo ""
	@echo "$(YELLOW)Code Quality:$(RESET)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "check" "Run ruff code check"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "format" "Run ruff code formatter"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "format-check" "Check if code format meets standards"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "lint" "Run complete code quality check (check + format verification)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "fix" "Automatically fix code format and fixable issues"
	@echo ""
	@echo "$(YELLOW)Project Management:$(RESET)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "install" "Install project dependencies"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "sync" "Sync dependencies (equivalent to install)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "add" "Add new dependency (usage: make add PKG=package_name)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "clean" "Clean temporary files and cache"
	@echo ""
	@echo "$(YELLOW)Development and Execution:$(RESET)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "run" "Run main program (audio processing pipeline)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "run-preview" "Run preview mode (process only first 10 minutes)"
	@echo ""
	@echo "$(YELLOW)Utility Tools:$(RESET)"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "show-videos" "Display video files directory"
	@printf "  $(GREEN)%-15s$(RESET) %s\n" "info" "Display project information"

# =============================================================================
# Code Quality
# =============================================================================

.PHONY: check
check: ## Run ruff code check
	@echo "$(YELLOW)Running code quality check...$(RESET)"
	$(UVX) ruff check

.PHONY: format
format: ## Run ruff code formatter
	@echo "$(YELLOW)Formatting code...$(RESET)"
	$(UVX) ruff format

.PHONY: format-check
format-check: ## Check if code format meets standards
	@echo "$(YELLOW)Checking code format...$(RESET)"
	$(UVX) ruff format --check

.PHONY: lint
lint: check format-check ## Run complete code quality check (check + format verification)
	@echo "$(GREEN)✓ Code quality check completed$(RESET)"

.PHONY: fix
fix: format check ## Automatically fix code format and fixable issues
	@echo "$(GREEN)✓ Code fixes completed$(RESET)"

# =============================================================================
# Project Management
# =============================================================================

.PHONY: install
install: ## Install project dependencies
	@echo "$(YELLOW)Installing project dependencies...$(RESET)"
	$(UV) sync

.PHONY: sync
sync: ## Sync dependencies (equivalent to install)
	@echo "$(YELLOW)Syncing project dependencies...$(RESET)"
	$(UV) sync

.PHONY: add
add: ## Add new dependency (usage: make add PKG=package_name)
	@if [ -z "$(PKG)" ]; then \
		echo "$(RED)Error: Please specify package name$(RESET)"; \
		echo "Usage: make add PKG=package_name"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Adding dependency: $(PKG)$(RESET)"
	$(UV) add $(PKG)

.PHONY: clean
clean: ## Clean temporary files and cache
	@echo "$(YELLOW)Cleaning temporary files...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaning completed$(RESET)"

# =============================================================================
# Development and Execution
# =============================================================================

.PHONY: run
run: ## Run main program (audio processing pipeline)
	@echo "$(YELLOW)Running audio processing pipeline...$(RESET)"
	$(UV) run python $(SCRIPTS_DIR)/extract.py \
		-d $(VIDEOS_DIR) -r $(RUNS_DIR) \
		--reuse --log-level INFO \
		--separate-vocals --demucs-model mdx_q --demucs-device mps --demucs-two-stems vocals \
		--vad-aggr 2 --vad-min-ms 150 --vad-merge-ms 200 \
		--score-thr 0.15 --min-sec 0.20 --merge-gap 0.30 --pad-sec 0.20 \
		--use-panns --panns-thr 0.05 \
		--extract-quirks --quirks-panns --quirks-panns-thr 0.30 \
		--gen-subs --subs-language zh


.PHONY: run-preview
run-preview: ## Run preview mode (process only first 10 minutes)
	@echo "$(YELLOW)Running preview mode (first 10 minutes)...$(RESET)"
	$(UV) run python $(SCRIPTS_DIR)/extract.py \
		-d $(VIDEOS_DIR) -r $(RUNS_DIR) \
		--reuse --log-level INFO \
		--preview-minutes 10 \
		--separate-vocals --demucs-model mdx_q --demucs-device mps --demucs-two-stems vocals \
		--vad-aggr 2 --vad-min-ms 150 --vad-merge-ms 200 \
		--score-thr 0.15 --min-sec 0.20 --merge-gap 0.30 --pad-sec 0.20 \
		--gen-subs --subs-language zh


# =============================================================================
# Utility Tools
# =============================================================================

.PHONY: show-videos
show-videos: ## Display video files directory
	@echo "$(YELLOW)Video files directory:$(RESET)"
	@if [ -d "$(VIDEOS_DIR)" ]; then \
		ls -la $(VIDEOS_DIR)/*.mp4 2>/dev/null || echo "$(RED)No .mp4 files found$(RESET)"; \
	else \
		echo "$(RED)Videos directory does not exist: $(VIDEOS_DIR)$(RESET)"; \
	fi

.PHONY: info
info: ## Display project information
	@echo "$(BOLD)Project Information$(RESET)"
	@echo "Project name: KANA"
	@echo "Python version: $(shell python --version 2>/dev/null || echo 'Not installed')"
	@echo "UV version: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo "Project path: $(shell pwd)"
	@echo "Videos directory: $(VIDEOS_DIR)"
	@echo "Runs directory: $(RUNS_DIR)"
