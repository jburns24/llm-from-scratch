POETRY := poetry
PYTHON := $(POETRY) run python
DATA_DIR := ./data

.PHONY: all install clean run test lint format teardown

# Default target
all: install run

# Install dependencies using poetry
install: set-env
	$(POETRY) install --no-root

set-env:
	$(POETRY) env use 3.11

# Run the application
run: install
	@DATA_DIR=$(DATA_DIR) $(PYTHON) ./main.py

# Clean up compiled Python files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete
	find . -type d -name ".pytest_cache" -delete

# Run tests
test:
	$(POETRY) run pytest

# Run linting
lint:
	$(POETRY) run pylint src/

# Format code
format:
	$(POETRY) run black src/

# Remove poetry environment
teardown:
	$(POETRY) env remove --all

# Full cleanup
wipe: clean teardown
