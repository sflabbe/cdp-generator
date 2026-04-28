UV ?= uv
PYTHONPATHS := cdp_generator tests

.PHONY: sync test lint format format-check lock lock-check run-cdp run-steel clean

sync:
	$(UV) sync --all-extras --dev

test:
	$(UV) run pytest

lint:
	$(UV) run flake8 $(PYTHONPATHS)

format:
	$(UV) run isort $(PYTHONPATHS)
	$(UV) run black $(PYTHONPATHS)

format-check:
	$(UV) run isort --check-only $(PYTHONPATHS)
	$(UV) run black --check $(PYTHONPATHS)

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
