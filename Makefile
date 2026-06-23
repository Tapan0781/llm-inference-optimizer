.PHONY: setup-local lint test test-all format clean

setup-local:
	python -m pip install --upgrade pip
	pip install -r requirements/base.txt -r requirements/dev.txt
	pip install -e .

lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
