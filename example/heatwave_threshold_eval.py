import os
import gcsfs
import xarray as xr
import pandas as pd
import numpy as np
import extremeweatherbench as ewb
from cleandata import cleanData

print("Step 0: Initializing HPC Data Processor & GCS connection...")

scratch_dir = os.environ.get('SCRATCH_hai_1247')
forecast_zarr_path = f"{scratch_dir}/beast_forecast/latest_damien_heatwaves.zarr/events_heat_wave_1/"
target_zarr_path = f"{scratch_dir}/era5_2020_2025_37levels.zarr"
print("forecast_zarr_path: ", forecast_zarr_path)


# ---------------------------------------------------------
# Step 1: Load the Forecast, Compute Valid Time, & Clean Data
# ---------------------------------------------------------
print("Loading forecast dataset...")
ds_forecast = xr.open_dataset(forecast_zarr_path, engine="zarr", consolidated=False)

required_vars = ["2m_temperature"]
clean_ds = cleanData(ds_forecast, required_vars)

# --- Compute Valid Time from init_time + lead_time ---
print("Deriving valid time from init_time + lead_time...")
# Xarray broadcasts init_time (1D) and lead_time (1D) across dimensions
valid_times = clean_ds["init_time"] + clean_ds["lead_time"]
print("Valid time: ",valid_times[0][0])

# Create the EWB forecast object
my_forecast = ewb.inputs.XarrayForecast(
    ds=clean_ds,  
    name="My_HPC_Model",
    variables=required_vars,
)

print("Loading actual ground truth dataset for the metric...")
inhouse_target = ewb.inputs.ERA5(
    source=target_zarr_path,  
    variables=required_vars,
    variable_mapping={"time": "valid_time"}
)

# ---------------------------------------------------------
# Step 2: Load and Align Google's ERA5 Climatology Target
# ---------------------------------------------------------
print("Connecting to Google GCS hourly climatology via fsspec mapper...")

fs = gcsfs.GCSFileSystem(token="anon")
gcs_climatology_path = 'weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr'
mapper = fs.get_mapper(gcs_climatology_path)

ds_clim_raw = xr.open_dataset(mapper, engine="zarr")

clim_subset = ds_clim_raw[required_vars]

print("Extracting unique valid times to build a standard target valid_time series...")
flat_valid_times = valid_times.values.flatten()
unique_valid_times = pd.unique(pd.to_datetime(flat_valid_times))
unique_valid_times = np.sort(unique_valid_times)

# --- SOLUTION: Use 'valid_time' as the dimension name instead of 'time' ---
time_da = xr.DataArray(
    unique_valid_times, 
    dims=["valid_time"], 
    coords={"valid_time": unique_valid_times}
)

print("Aligning climatology coordinates to unique valid times...")
target_aligned = clim_subset.sel(
    dayofyear=time_da.dt.dayofyear,
    hour=time_da.dt.hour,
    method="nearest"
)

target_aligned = target_aligned.drop_vars(["dayofyear", "hour"], errors="ignore")

temp_nc_path = f"{scratch_dir}/beast_forecast/google_climatology_temp.nc"
temp_zarr_path = f"{scratch_dir}/beast_forecast/google_climatology_target.zarr"

print(f"Step A: Writing temporary NetCDF: {temp_nc_path}")
target_aligned.to_netcdf(temp_nc_path)

print(f"Step B: Exporting clean target Zarr store: {temp_zarr_path}")
ds_clean = xr.open_dataset(temp_nc_path)

for var in ds_clean.variables:
    ds_clean[var].encoding.clear()

ds_clean.to_zarr(temp_zarr_path, mode="w")

# Create the official EWB Target object mapped to 'valid_time'
# climatology_temperature_da = ewb.inputs.ERA5(
#     source=temp_zarr_path,
#     variables=required_vars,
#     variable_mapping={"valid_time": "valid_time"}
# )
print("Loading climatology DataArray for baseline anomaly calculation...")
ds_climatology = xr.open_dataset(temp_zarr_path, engine="zarr")
climatology_temperature_da = ds_climatology["2m_temperature"]
# ---------------------------------------------------------
# Step 3: Define Event Types & Metrics
# ---------------------------------------------------------
event_type_name = "heat_wave"  

eval_objects = [
    ewb.inputs.EvaluationObject(
        event_type=event_type_name, 
        metric_list=[
            
            ewb.metrics.EnsembleHeatWaveDetectionRate(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature",
                climatology=climatology_temperature_da,
            ),
            ewb.metrics.EnsembleHeatWaveFalseAlarmRate(
                forecast_variable="2m_temperature",
                target_variable="2m_temperature",
                climatology=climatology_temperature_da,
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

heatwave_cases = [c for c in all_cases_db if c.event_type == "heat_wave"]
heatwave_case_1 = [c for c in heatwave_cases if c.case_id_number == 1]

print(f"Found {len(heatwave_case_1)} heatwave cases. Running evaluation pipeline...")

runner = ewb.ExtremeWeatherBench(
    case_metadata=heatwave_case_1, 
    evaluation_objects=eval_objects,
    selected_ensemble=50
)

outputs = runner.run_evaluation()

# ---------------------------------------------------------
# Step 5: Save and Verify Results
# ---------------------------------------------------------
csv_filename = f"csv/heatwave_threshold_evaluation_results_{heatwave_case_1[0].case_id_number}.csv"

if not outputs.empty:
    outputs.to_csv(csv_filename, index=False)
    print(f"\nEvaluation Complete! Results successfully saved to {csv_filename}")
    print("\n--- Top Results ---")
    print(outputs.head())
else:
    print("\nWarning: Evaluation returned empty results. Spatial/temporal dimensions might not have matched in the pipeline.")