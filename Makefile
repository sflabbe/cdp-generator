UV ?= uv
PYTHONPATHS := cdp_generator tests

.PHONY: sync test lint typecheck format format-check check lock lock-check run-cdp run-steel clean

sync:
	$(UV) sync --all-extras --dev

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check $(PYTHONPATHS)

typecheck:
	$(UV) run mypy cdp_generator

format:
	$(UV) run ruff check --fix $(PYTHONPATHS)
	$(UV) run ruff format $(PYTHONPATHS)

format-check:
	$(UV) run ruff format --check $(PYTHONPATHS)

check: lint typecheck format-check test

lock:
	$(UV) lock

lock-check:
	$(UV) lock --check

run-cdp:
	$(UV) run cdp-generator

run-steel:
	$(UV) run cdp-steel

clean:
	rm -rf .venv .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
