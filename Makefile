UV ?= uv
RUN = $(UV) run --locked --no-sync

.PHONY: setup lock lint format types security test bench bench-compare quality ci build kg-schemas-check
setup:
	$(UV) python install
	$(UV) sync --locked --group dev
lock:
	$(UV) lock --check
lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests
format:
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests
types:
	$(RUN) mypy src/goes_natural_science_kg
security:
	$(RUN) bandit -q -r src
	$(RUN) pip-audit --skip-editable
test:
	$(RUN) pytest --benchmark-disable --cov=goes_natural_science_kg --cov-report=term-missing
bench:
	$(RUN) pytest tests/benchmarks --benchmark-only --benchmark-json=reports/benchmark.json
# Reference and current runs must use comparable hardware and Python versions.
bench-compare:
	$(RUN) pytest tests/benchmarks --benchmark-only --benchmark-compare=tests/benchmarks/baseline.json --benchmark-compare-fail=mean:20%
kg-schemas-check:
	$(RUN) pytest tests/unit/test_repository.py --benchmark-disable
quality: lint types test
ci: lock quality
	$(RUN) bandit -q -r src
build:
	$(UV) build
