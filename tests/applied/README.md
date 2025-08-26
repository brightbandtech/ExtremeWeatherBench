# Miniaturized Applied Tests

This directory contains miniaturized versions of the applied scripts from `docs/examples/` designed for fast verification of the ExtremeWeatherBench pipeline.

## Purpose

These scripts provide a quick way to test that the EWB system is working correctly without the long execution times of the full applied scripts. They use:

- Single cases instead of multiple cases
- Reduced chunking for faster data loading
- Essential metrics only
- Smaller tolerance ranges
- Optimized data access patterns

## Scripts

### Individual Tests

- `mini_applied_ar.py` - Atmospheric river evaluation with one case
- `mini_applied_heatwave.py` - Heat wave evaluation with one case  
- `mini_applied_tc.py` - Tropical cyclone evaluation with one case

### Combined Test

- `mini_applied_all.py` - Runs all three event types sequentially with timing and error reporting

## Usage

### Via Makefile (Recommended)

```bash
# Run all miniaturized tests
make test-mini

# Run individual tests
make test-mini-ar       # Atmospheric rivers only
make test-mini-heatwave # Heat waves only  
make test-mini-tc       # Tropical cyclones only
```

### Direct Execution

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all tests
python tests/applied/mini_applied_all.py

# Or run individual tests
python tests/applied/mini_applied_ar.py
python tests/applied/mini_applied_heatwave.py
python tests/applied/mini_applied_tc.py
```

## Performance

These scripts are designed to complete in **minutes rather than hours**, making them suitable for:

- CI/CD pipeline verification
- Quick development testing
- Confirming system setup
- Debugging basic functionality

## Data Requirements

The scripts use the same data sources as the full applied scripts but with reduced scope:

- **ERA5**: ARCO ERA5 full dataset (public)
- **HRES**: WeatherBench2 HRES forecasts (public)
- **GHCN**: Default GHCN station data
- **IBTrACS**: Tropical cyclone best track data

### Authentication

These scripts require Google Cloud authentication to access the public datasets. Before running the tests, ensure you have authenticated:

```bash
# Install gcloud CLI if not already installed
# https://cloud.google.com/sdk/docs/install

# Authenticate for application default credentials
gcloud auth application-default login
```

If you see authentication errors like "Reauthentication is needed", run the `gcloud auth application-default login` command.

## Expected Output

Each script will:

1. Load and filter case data to 1-2 cases maximum
2. Set up optimized data sources with chunking
3. Run evaluation with reduced tolerance
4. Print sample results and timing information
5. Exit with code 0 on success, 1 on failure

The combined script (`mini_applied_all.py`) provides a comprehensive test report with per-test timing and overall success/failure status.
