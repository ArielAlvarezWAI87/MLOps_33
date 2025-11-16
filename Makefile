# ============================================================================
# Makefile - Steel Energy Consumption Prediction MLOps Pipeline
# ============================================================================
#
# Usage:
#   make help          - Show all available commands
#   make setup         - Initialize project and DVC
#   make snapshot      - Create data snapshots
#   make pipeline      - Run full pipeline
#   make train         - Train model only
#   make clean         - Clean temporary files
# ============================================================================

.PHONY: help setup snapshot validate pipeline train predict metrics plots clean install test

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Project directories
PROJECT_ROOT := $(shell pwd)
DATA_DIR := $(PROJECT_ROOT)/data
SRC_DIR := $(PROJECT_ROOT)/src
MODELS_DIR := $(PROJECT_ROOT)/models
LOGS_DIR := $(PROJECT_ROOT)/logs

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║         Steel Energy Prediction - MLOps Pipeline Commands          ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Initialization:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(setup|install|init-dvc)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Data Management:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(snapshot|validate|pull|push)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Pipeline Execution:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(pipeline|preprocess|features|train|evaluate|predict)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Analysis & Monitoring:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(metrics|plots|status|dag|logs)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(test|lint|format|clean)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# SETUP & INITIALIZATION
# ============================================================================

setup: install init-dvc ## Complete project setup (install deps + init DVC)
	@echo "$(GREEN)✓ Project setup complete!$(NC)"

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	pip install dvc pytest pytest-cov black flake8
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

init-dvc: ## Initialize DVC in the project
	@echo "$(BLUE)Initializing DVC...$(NC)"
	python dataset_snapshot.py --init
	@echo "$(GREEN)✓ DVC initialized$(NC)"

# ============================================================================
# DATA MANAGEMENT
# ============================================================================

snapshot: ## Create snapshots of all datasets
	@echo "$(BLUE)Creating data snapshots...$(NC)"
	python dataset_snapshot.py --snapshot all
	@echo "$(GREEN)✓ Snapshots created$(NC)"

snapshot-raw: ## Create snapshot of raw data only
	@echo "$(BLUE)Creating raw data snapshot...$(NC)"
	python dataset_snapshot.py --snapshot raw
	@echo "$(GREEN)✓ Raw data snapshot created$(NC)"

snapshot-processed: ## Create snapshot of processed data only
	@echo "$(BLUE)Creating processed data snapshot...$(NC)"
	python dataset_snapshot.py --snapshot processed
	@echo "$(GREEN)✓ Processed data snapshot created$(NC)"

snapshot-features: ## Create snapshot of feature matrices
	@echo "$(BLUE)Creating features snapshot...$(NC)"
	python dataset_snapshot.py --snapshot features
	@echo "$(GREEN)✓ Features snapshot created$(NC)"

validate: ## Validate data integrity
	@echo "$(BLUE)Validating data integrity...$(NC)"
	python dataset_snapshot.py --validate
	@echo "$(GREEN)✓ Validation complete$(NC)"

list-snapshots: ## List all recorded snapshots
	@echo "$(BLUE)Recorded snapshots:$(NC)"
	python dataset_snapshot.py --list

pull: ## Pull data from DVC remote
	@echo "$(BLUE)Pulling data from remote...$(NC)"
	dvc pull
	@echo "$(GREEN)✓ Data pulled$(NC)"

push: ## Push data to DVC remote
	@echo "$(BLUE)Pushing data to remote...$(NC)"
	dvc push
	@echo "$(GREEN)✓ Data pushed$(NC)"

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

pipeline: ## Run complete ML pipeline
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                    Running Complete Pipeline                       ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════════╝$(NC)"
	dvc repro
	@echo "$(GREEN)✓ Pipeline complete!$(NC)"

pipeline-force: ## Force rerun entire pipeline (ignore cache)
	@echo "$(BLUE)Force running pipeline...$(NC)"
	dvc repro -f
	@echo "$(GREEN)✓ Pipeline complete!$(NC)"

preprocess: ## Run preprocessing stage only
	@echo "$(BLUE)Running preprocessing...$(NC)"
	dvc repro -s preprocessing
	@echo "$(GREEN)✓ Preprocessing complete$(NC)"

features: ## Run feature engineering stage only
	@echo "$(BLUE)Running feature engineering...$(NC)"
	dvc repro -s feature_engineering
	@echo "$(GREEN)✓ Feature engineering complete$(NC)"

train: ## Run model training stage only
	@echo "$(BLUE)Training model...$(NC)"
	dvc repro -s train_rulefit
	@echo "$(GREEN)✓ Model training complete$(NC)"

evaluate: ## Run model evaluation stage
	@echo "$(BLUE)Evaluating model...$(NC)"
	dvc repro -s evaluate
	@echo "$(GREEN)✓ Model evaluation complete$(NC)"

predict: ## Run prediction stage
	@echo "$(BLUE)Generating predictions...$(NC)"
	dvc repro -s predict
	@echo "$(GREEN)✓ Predictions generated$(NC)"

# ============================================================================
# ANALYSIS & MONITORING
# ============================================================================

status: ## Check pipeline status
	@echo "$(BLUE)Pipeline status:$(NC)"
	dvc status

dag: ## Show pipeline dependency graph
	@echo "$(BLUE)Pipeline DAG:$(NC)"
	dvc dag

metrics: ## Show current metrics
	@echo "$(BLUE)Current metrics:$(NC)"
	dvc metrics show

metrics-diff: ## Compare metrics with previous commit
	@echo "$(BLUE)Metrics comparison:$(NC)"
	dvc metrics diff

params: ## Show current parameters
	@echo "$(BLUE)Current parameters:$(NC)"
	dvc params show

params-diff: ## Compare parameters with previous commit
	@echo "$(BLUE)Parameters comparison:$(NC)"
	dvc params diff

plots: ## Show plots
	@echo "$(BLUE)Generating plots...$(NC)"
	dvc plots show
	@echo "$(GREEN)✓ Plots generated in dvc_plots/$(NC)"

plots-diff: ## Compare plots with previous commit
	@echo "$(BLUE)Comparing plots...$(NC)"
	dvc plots diff

logs: ## Show recent pipeline logs
	@echo "$(BLUE)Recent logs:$(NC)"
	@tail -n 50 $(LOGS_DIR)/pipeline.log || echo "No logs found"

mlflow-ui: ## Launch MLflow UI
	@echo "$(BLUE)Starting MLflow UI on http://localhost:5000$(NC)"
	mlflow ui --port 5000

# ============================================================================
# TESTING & QUALITY
# ============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-preprocessing: ## Run preprocessing tests only
	@echo "$(BLUE)Testing preprocessing...$(NC)"
	pytest tests/test_preprocessing.py -v
	@echo "$(GREEN)✓ Preprocessing tests complete$(NC)"

test-features: ## Run feature engineering tests only
	@echo "$(BLUE)Testing feature engineering...$(NC)"
	pytest tests/test_feature_engineering.py -v
	@echo "$(GREEN)✓ Feature tests complete$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

lint: ## Run code linting
	@echo "$(BLUE)Linting code...$(NC)"
	flake8 src/ tests/ --max-line-length=100
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking code format...$(NC)"
	black --check src/ tests/
	@echo "$(GREEN)✓ Format check complete$(NC)"

# ============================================================================
# CLEANUP
# ============================================================================

clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf dist/
	rm -rf build/
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-data: ## Clean all data files (WARNING: removes all data)
	@echo "$(RED)WARNING: This will remove all data files!$(NC)"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read confirm
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/tmp/*
	@echo "$(YELLOW)✓ Data files removed$(NC)"

clean-models: ## Clean all model files
	@echo "$(BLUE)Cleaning model files...$(NC)"
	rm -rf $(MODELS_DIR)/*.pkl
	@echo "$(GREEN)✓ Model files removed$(NC)"

clean-logs: ## Clean log files
	@echo "$(BLUE)Cleaning logs...$(NC)"
	rm -rf $(LOGS_DIR)/*.log
	@echo "$(GREEN)✓ Logs cleaned$(NC)"

clean-all: clean clean-models clean-logs ## Clean everything (data preserved)
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

reproduce: ## Reproduce pipeline from scratch
	@echo "$(BLUE)Reproducing pipeline from scratch...$(NC)"
	$(MAKE) clean
	dvc pull
	dvc repro -f
	@echo "$(GREEN)✓ Pipeline reproduced$(NC)"

checkout-version: ## Checkout specific Git/DVC version (usage: make checkout-version COMMIT=abc123)
	@echo "$(BLUE)Checking out version $(COMMIT)...$(NC)"
	git checkout $(COMMIT)
	dvc checkout
	@echo "$(GREEN)✓ Version checked out$(NC)"

# ============================================================================
# DOCKER (if applicable)
# ============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t steel-energy-predictor:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run pipeline in Docker
	@echo "$(BLUE)Running pipeline in Docker...$(NC)"
	docker run --rm -v $(PROJECT_ROOT):/app steel-energy-predictor:latest make pipeline
	@echo "$(GREEN)✓ Docker run complete$(NC)"

# ============================================================================
# GIT + DVC WORKFLOW
# ============================================================================

commit-pipeline: ## Commit pipeline changes (Git + DVC)
	@echo "$(BLUE)Committing pipeline changes...$(NC)"
	git add dvc.yaml params.yaml dvc.lock
	git commit -m "Update pipeline configuration"
	dvc push
	git push
	@echo "$(GREEN)✓ Pipeline changes committed$(NC)"

commit-data: ## Commit data changes
	@echo "$(BLUE)Committing data changes...$(NC)"
	git add data/*.dvc
	git commit -m "Update tracked datasets"
	dvc push
	git push
	@echo "$(GREEN)✓ Data changes committed$(NC)"

sync: ## Sync with remote (Git pull + DVC pull)
	@echo "$(BLUE)Syncing with remote...$(NC)"
	git pull
	dvc pull
	@echo "$(GREEN)✓ Sync complete$(NC)"

# ============================================================================
# UTILITIES
# ============================================================================

check-env: ## Check if environment is correctly set up
	@echo "$(BLUE)Checking environment...$(NC)"
	@python --version
	@pip --version
	@dvc version
	@git --version
	@echo "$(GREEN)✓ Environment OK$(NC)"

create-dirs: ## Create necessary directories
	@echo "$(BLUE)Creating directories...$(NC)"
	mkdir -p $(DATA_DIR)/raw
	mkdir -p $(DATA_DIR)/processed
	mkdir -p $(DATA_DIR)/tmp
	mkdir -p $(MODELS_DIR)
	mkdir -p $(LOGS_DIR)
	mkdir -p predictions
	mkdir -p metrics
	mkdir -p plots
	@echo "$(GREEN)✓ Directories created$(NC)"

# ============================================================================
# DEVELOPMENT
# ============================================================================

notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs: ## Show DVC instructions
	@echo "$(BLUE)DVC Workflow Instructions:$(NC)"
	python dataset_snapshot.py --instructions

# ============================================================================
# Quick Commands (shortcuts)
# ============================================================================

r: pipeline ## Shortcut for 'make pipeline'
t: train ## Shortcut for 'make train'
p: predict ## Shortcut for 'make predict'
m: metrics ## Shortcut for 'make metrics'
s: status ## Shortcut for 'make status'

# ============================================================================
# END
# ============================================================================