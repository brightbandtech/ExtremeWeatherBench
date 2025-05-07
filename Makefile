.PHONY: help run default test clean

CLI=ewb

help:
	@echo "Available commands:"
	@echo "  make run [ARGS='']      - Run the CLI with custom arguments"
	@echo "  make default            - Run the CLI with --default"
	@echo "  make test               - Run a basic test of the CLI"
	@echo "  make clean              - Remove output files"

run:
	$(CLI) $(ARGS)

default:
	$(CLI) --default

test:
	# Example: test with default config and check output
	$(CLI) --default
	@if [ -f outputs/evaluation_results.csv ]; then \
		echo "Test passed: Output file created."; \
	else \
		echo "Test failed: Output file not found!"; \
		exit 1; \
	fi

clean:
	rm -rf outputs/evaluation_results.csv 