PY ?= $(shell command -v python3.12 || command -v python3.11 || echo python3)
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help install run run-h5 refresh test lint typecheck check clean

help:
	@echo "make install    create $(VENV) and install the package with dev tools"
	@echo "make run        full walk-forward backtest (h=1) -> results/ figures/"
	@echo "make run-h5     5-day-ahead variant -> results_h5/ figures_h5/"
	@echo "make refresh    re-download VIX history, then run"
	@echo "make test       pytest"
	@echo "make lint       ruff (lint + format check)"
	@echo "make typecheck  mypy --strict"
	@echo "make check      lint + typecheck + test"

$(BIN)/vixcast:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

install: $(BIN)/vixcast

run: install
	$(BIN)/vixcast

run-h5: install
	$(BIN)/vixcast --horizon 5 --results-dir results_h5 --figures-dir figures_h5

refresh: install
	$(BIN)/vixcast --refresh-data

test: install
	$(BIN)/pytest

lint: install
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

typecheck: install
	$(BIN)/mypy

check: lint typecheck test

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
