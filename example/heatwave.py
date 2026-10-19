
import os
import pathlib
import extremeweatherbench as ewb
import xarray as xr
import pandas as pd
from cleandata import cleanData

print("Step 0: Initializing HPC Data Processor...")

# Define paths for target and forecast datasets
scratch_dir = os.environ.get('SCRATCH_hai_1247')
forecast_zarr_path = f"{scratch_dir}/beast_forecast/latest_damien_heatwaves.zarr/events_heat_wave_1/"
# forecast_zarr_path = f"{scratch_dir}/beast_forecast/forecasts.zarr/events_heat_wave_1/"
target_zarr_path = f"{scratch_dir}/era5_2020_2025_37levels.zarr"
print("forecast_zarr_path: ",forecast_zarr_path)

# ---------------------------------------------------------
# Step 1: Load the Forecast & Clean Data
# ---------------------------------------------------------
print("Loading forecast dataset...")
# Using consolidated=False can speed up Zarr file reading on HPC systems
ds_forecast = xr.open_dataset(forecast_zarr_path, engine="zarr", consolidated=False)

required_vars = ["2m_temperature"]
clean_ds = cleanData(ds_forecast, required_vars)
print(clean_ds)

# Create the EWB forecast object
# Note: Passing the cleaned dataset (clean_ds) instead of the raw one
my_forecast = ewb.inputs.XarrayForecast(
    ds=clean_ds,  
    name="My_HPC_Model",
    variables=required_vars,
    # No variable mapping is needed since '2m_temperature' is used in both datasets
)

# ---------------------------------------------------------
# Step 2: Load the Target
# ---------------------------------------------------------
print("Loading target dataset...")
inhouse_target = ewb.inputs.ERA5(
    source=target_zarr_path,
    variables=required_vars,
    variable_mapping={"time": "valid_time"}
)

# ---------------------------------------------------------
# Step 3: Define Event Types & Metrics
# ---------------------------------------------------------
# The standard event_type name in EWB is "heatwave"
event_type_name = "heat_wave"  

eval_objects = [
    ewb.inputs.EvaluationObject(
        event_type=event_type_name, 
        metric_list=[
            ewb.metrics.SPEDS(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature"   
            ),
            ewb.metrics.SpatialFSS(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature",
                threshold = 37.0
            ),
            ewb.metrics.CoverOneSD(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature"
            ),
            ewb.metrics.CRPS(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature"
            ),
            ewb.metrics.NLL(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature"
            ),
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature"
            )
        ],
        target=inhouse_target,
        forecast=my_forecast,
    )
]

# ---------------------------------------------------------
# Step 4: Run the Evaluation
# ---------------------------------------------------------
print("Loading EWB's default database of extreme events...")
all_cases_db = ewb.cases.load_ewb_events_yaml_into_case_list()

# Filter ONLY Heatwave cases to avoid running the evaluation on cyclones or coldwaves
heatwave_cases = [c for c in all_cases_db if c.event_type == "heat_wave"]
heatwave_case_1 = [c for c in heatwave_cases if c.case_id_number == 1]

# (Optional) Uncomment the line below if you only want to evaluate a specific Heatwave Case ID
# heatwave_cases = [c for c in heatwave_cases if c.case_id_number == 1]

print(f"Found {len(heatwave_case_1)} heatwave cases. Running evaluation pipeline...")

# Initialize the runner with the correct ExtremeWeatherBench class
runner = ewb.evaluate.ExtremeWeatherBench(
    case_metadata=heatwave_case_1, 
    evaluation_objects=eval_objects,
    selected_ensemble=10
)

outputs = runner.run_evaluation()

# ---------------------------------------------------------
# Step 5: Save and Verify Results
# ---------------------------------------------------------
# Get the directory where the current script (heatwave.py) is located
script_dir = pathlib.Path(__file__).resolve().parent

# Build a robust path pointing to 'ExtremeWeatherBench/example/csv'
output_dir = script_dir / "csv"
output_dir.mkdir(parents=True, exist_ok=True)  # Automatically creates the folder if missing

# Define the final CSV filename path
csv_filename = output_dir / f"heatwave_evaluation_results_{heatwave_case_1[0].case_id_number}.csv"

# Check if the resulting dataframe is not empty before saving
if not outputs.empty:
    outputs.to_csv(csv_filename, index=False)
    print(f"\nEvaluation Complete! Results successfully saved to {csv_filename}")
    print("\n--- Top Results ---")
    print(outputs.head())
else:
    print("\nWarning: Evaluation returned empty results. Spatial/temporal dimensions might not have matched in the pipeline.")