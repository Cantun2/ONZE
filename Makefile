# ONZE — task runner. All Python entrypoints run as modules from the repo root
# so `src` is importable (see src/__init__.py). Some entrypoints don't exist
# yet; targets are wired to their intended paths per ONZE_spec.md §4.2.

PYTHON ?= python3
PIP ?= pip3
UVICORN ?= uvicorn

.DEFAULT_GOAL := help
.PHONY: help install ingest ratings fit backtest api test frontend all

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PIP) install -r requirements.txt

ingest: ## Load + harmonize raw datasets (spec §2)
	$(PYTHON) -m src.ingest.loaders

ratings: ## Compute Elo walk-forward history (spec §3.1)
	$(PYTHON) -m src.ratings.elo

fit: ## Fit Dixon-Coles goal model (spec §3.2, §3.3)
	$(PYTHON) -m src.model.dixon_coles

backtest: ## Walk-forward backtest vs baselines (spec §3.6, §3.7)
	$(PYTHON) -m src.eval.backtest

api: ## Serve the FastAPI app (spec §4.4)
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run the test suite
	$(PYTHON) -m pytest

frontend: ## Run the React/Vite dev server (spec §5)
	cd frontend && npm run dev

all: ingest ratings fit backtest ## Full offline pipeline: ingest -> ratings -> fit -> backtest
