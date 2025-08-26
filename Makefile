.PHONY: help install run default test test-config test-parallel test-cache test-mini test-mini-ar test-mini-heatwave test-mini-tc verify-setup clean lint format typecheck dev-test all-tests

CLI=ewb
OUTPUT_DIR=make_outputs
TEST_CONFIG=docs/examples/example_config.py

help:
	@echo "ExtremeWeatherBench Makefile"
	@echo ""
	@echo "Installation:"
	@echo "  make install            - Install the package in development mode"
	@echo ""
	@echo "Basic Usage:"
	@echo "  make run [ARGS='']      - Run the CLI with custom arguments"
	@echo "  make default            - Run the CLI with --default mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test               - Run basic test with default mode (includes --precompute)"
	@echo "  make test-config        - Test with custom config file (includes --precompute)"
	@echo "  make test-parallel      - Test with parallel execution (4 jobs + --precompute)"
	@echo "  make test-cache         - Test with caching enabled (includes --precompute)"
	@echo "  make test-mini          - Run all miniaturized tests (fast verification)"
	@echo "  make test-mini-ar       - Run miniaturized atmospheric river test"
	@echo "  make test-mini-heatwave - Run miniaturized heatwave test"
	@echo "  make test-mini-tc       - Run miniaturized tropical cyclone test"
	@echo "  make verify-setup       - Verify setup without external data access"
	@echo "  make all-tests          - Run all test scenarios"
	@echo ""
	@echo "Development:"
	@echo "  make lint               - Run ruff linting"
	@echo "  make format             - Run ruff formatting"
	@echo "  make typecheck          - Run mypy type checking"
	@echo "  make dev-test           - Run pytest test suite"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - Remove output files and cache"

# Installation
install:
	@echo "Installing ExtremeWeatherBench in development mode..."
	pip install -e .

# Basic usage targets
run:
	$(CLI) $(ARGS)

default:
	@echo "Running EWB with default configuration..."
	$(CLI) --default --output-dir $(OUTPUT_DIR)/default

# Testing targets
test:
	@echo "Running basic test with default configuration..."
	$(CLI) --default --precompute --output-dir $(OUTPUT_DIR)/test
	@if [ -f $(OUTPUT_DIR)/test/evaluation_results.csv ]; then \
		echo "✓ Test passed: Output file created at $(OUTPUT_DIR)/test/evaluation_results.csv"; \
		ls -la $(OUTPUT_DIR)/test/; \
	else \
		echo "✗ Test failed: Output file not found!"; \
		exit 1; \
	fi

test-config:
	@echo "Testing with custom config file..."
	$(CLI) --config-file $(TEST_CONFIG) --precompute --output-dir $(OUTPUT_DIR)/config-test
	@if [ -f $(OUTPUT_DIR)/config-test/evaluation_results.csv ]; then \
		echo "✓ Config test passed: Output file created"; \
	else \
		echo "✗ Config test failed: Output file not found!"; \
		exit 1; \
	fi

test-parallel:
	@echo "Testing with parallel execution (4 jobs)..."
	$(CLI) --default --parallel 4 --precompute --output-dir $(OUTPUT_DIR)/parallel-test
	@if [ -f $(OUTPUT_DIR)/parallel-test/evaluation_results.csv ]; then \
		echo "✓ Parallel test passed: Output file created"; \
	else \
		echo "✗ Parallel test failed: Output file not found!"; \
		exit 1; \
	fi

test-cache:
	@echo "Testing with caching enabled..."
	$(CLI) --default --precompute --cache-dir $(OUTPUT_DIR)/cache --output-dir $(OUTPUT_DIR)/cache-test
	@if [ -f $(OUTPUT_DIR)/cache-test/evaluation_results.csv ]; then \
		echo "✓ Cache test passed: Output file created"; \
		if [ -d $(OUTPUT_DIR)/cache ]; then \
			echo "✓ Cache directory created"; \
		fi; \
	else \
		echo "✗ Cache test failed: Output file not found!"; \
		exit 1; \
	fi

test-save-operators:
	@echo "Testing case operator saving..."
	$(CLI) --default --precompute --save-case-operators $(OUTPUT_DIR)/case_operators.pkl --output-dir $(OUTPUT_DIR)/save-test
	@if [ -f $(OUTPUT_DIR)/case_operators.pkl ]; then \
		echo "✓ Case operators saved successfully"; \
	else \
		echo "✗ Failed to save case operators"; \
		exit 1; \
	fi

# Miniaturized test targets for fast verification
test-mini:
	@echo "Running all miniaturized tests..."
	@echo "Note: These tests require Google Cloud authentication for data access"
	@echo "Run 'gcloud auth application-default login' if you see authentication errors"
	.venv/bin/python tests/applied/mini_applied_all.py
	@echo "✓ All miniaturized tests completed!"

test-mini-ar:
	@echo "Running miniaturized atmospheric river test..."
	@echo "Note: Requires Google Cloud authentication for data access"
	.venv/bin/python tests/applied/mini_applied_ar.py
	@echo "✓ Atmospheric river test completed!"

test-mini-heatwave:
	@echo "Running miniaturized heatwave test..."
	@echo "Note: Requires Google Cloud authentication for data access"
	.venv/bin/python tests/applied/mini_applied_heatwave.py
	@echo "✓ Heatwave test completed!"

test-mini-tc:
	@echo "Running miniaturized tropical cyclone test..."
	@echo "Note: Requires Google Cloud authentication for data access"
	.venv/bin/python tests/applied/mini_applied_tc.py
	@echo "✓ Tropical cyclone test completed!"

verify-setup:
	@echo "Verifying ExtremeWeatherBench setup..."
	@echo "This test does not require external data access"
	.venv/bin/python tests/applied/verify_setup.py
	@echo "✓ Setup verification completed!"

all-tests: test test-config test-parallel test-cache test-save-operators verify-setup test-mini
	@echo "✓ All tests completed successfully!"

# Development targets
lint:
	@echo "Running ruff linting..."
	ruff check src/ tests/

format:
	@echo "Running ruff formatting..."
	ruff format src/ tests/

typecheck:
	@echo "Running mypy type checking..."
	mypy src/extremeweatherbench/

dev-test:
	@echo "Running pytest test suite..."
	pytest tests/ -v

# Cleanup
clean:
	@echo "Cleaning up output files and cache..."
	rm -rf $(OUTPUT_DIR)/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true 