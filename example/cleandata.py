import xarray as xr
import numpy as np
import pandas as pd
import extremeweatherbench as ewb
from extremeweatherbench import calc, utils, cases

def cleanData(
    era5_ds: xr.Dataset,
    required_vars: list = None
) -> xr.Dataset:
    """
    Cleans and restructures ERA5 dataset for Extreme Weatherbench.
    
    Args:
        era5_ds: The original xarray Dataset.
        required_vars: List of specific features to extract. If None, extracts all features.
    """
    
    # =====================================================================
    # STEP 1: Extract features dynamically based on required_vars
    # =====================================================================
    # era5_ds = era5_ds.mean(dim="ensemble")
    # print(era5_ds.coords["time"].values)
    # print("era5_ds--------------",era5_ds)
    available_features = list(era5_ds.feature.values)
    print("avail: ",available_features)
    
    if required_vars is not None:
        print(f"Extracting specified required variables: {required_vars}")
        # Validate if all requested variables exist in the dataset
        missing = [var for var in required_vars if var not in available_features]
        if missing:
            raise ValueError(f"Missing required variables: {missing}. Available: {available_features}")
        features_to_extract = required_vars
    else:
        print("No specific variables provided. Extracting ALL features...")
        features_to_extract = available_features

    # Get the name of the main data variable (usually 'fields')
    data_var_name = list(era5_ds.data_vars)[0]
    extracted_vars = {}
    
    # Iterate over the determined list of features
    for feature_name in features_to_extract:
        # Slice each feature and store it in the dictionary
        extracted_vars[feature_name] = era5_ds[data_var_name].sel(
            feature=feature_name
        ).drop_vars("feature", errors="ignore")
        
    # Create a new dataset with the extracted features
    modified_ds = xr.Dataset(extracted_vars)
    
    # Reassign all original coordinates (e.g., latitude, longitude, ensemble) except 'feature'
    for coord_name in era5_ds.coords:
        if coord_name != "feature":
            modified_ds = modified_ds.assign_coords({coord_name: era5_ds.coords[coord_name]})
            
    # =====================================================================
    # STEP 2: Time Restructuring Logic
    # =====================================================================
    
    # 1. Rename to match variables
    modified_ds = modified_ds.rename({"time": "valid_time"})
    
    # 2. Fix old lead_time format
    lead_time_hours = modified_ds.coords["lead_time"].values
    modified_ds = modified_ds.assign_coords(
        lead_time=lead_time_hours.astype("timedelta64[h]").astype("timedelta64[ns]")
    )
    print("lead", modified_ds.coords["lead_time"].values)
    # print("feature", modified_ds["mean_sea_level_pressure"].values)
    
    # 3. Calculate 1D Init Times
    earliest_prediction = min(modified_ds.coords["valid_time"].values)
    valid_time = modified_ds.valid_time.values
    old_lead_time = modified_ds.lead_time.values
    init_times_1d = earliest_prediction - old_lead_time  # Size: 4

    
    # 4. Calculate 2D Actual Lead Times 
    # Shape: (24, 4) -> (valid_time, old_lead_time)
    actual_lead_times_2d = valid_time[:, np.newaxis] - init_times_1d 

   # ---------------------------------------------------------
    # PRINT BLOCK FOR DEBUGGING ACTUAL DATA (IN HOURS)
    # ---------------------------------------------------------
    # Sirf print karne ke liye nanoseconds ko Hours (integer) mein convert karna
    old_lead_time_hr = old_lead_time.astype("timedelta64[h]").astype(int)
    actual_lead_times_2d_hr = actual_lead_times_2d.astype("timedelta64[h]").astype(int)


    
    # --- RESTRUCTURING STARTS HERE ---
    
    # A. Rename old dimensions to temporary names to avoid confusion
    modified_ds = modified_ds.rename(
        {"lead_time": "old_lead_dim", "valid_time": "old_valid_dim"}
    )
    
    # B. Assign newly calculated time coordinates as 2D variables
    modified_ds = modified_ds.assign_coords(
        actual_init_time=("old_lead_dim", init_times_1d),
        actual_lead_time=(("old_valid_dim", "old_lead_dim"), actual_lead_times_2d)
    )
    
    # C. STACK (Flatten the data into a 1D format)
    ds_stacked = modified_ds.stack(sample=("old_valid_dim", "old_lead_dim"))
    
    # D. Set the new actual time coordinates as the MultiIndex
    ds_stacked = ds_stacked.set_index(sample=["actual_init_time", "actual_lead_time"])
    
    # E. UNSTACK (Reshape back into a matrix - automatically handles diagonal shift & NaN filling)
    ds_final = ds_stacked.unstack("sample")
    
    # F. Final renaming for Extreme Weatherbench compatibility
    ds_final = ds_final.rename({
        "actual_init_time": "init_time", 
        "actual_lead_time": "lead_time"
    })

    #drop negative lead time
    ds_final = ds_final.where(ds_final.lead_time > np.timedelta64(0, 'ns'), drop=True)


    return ds_final

    

def prepare_era5_data_for_tc_detection(
    era5_ds: xr.Dataset,
    event_start: np.datetime64 | None = None,
    il_output: bool = False
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Prepare ERA5 data for TC track detection.
    
    Converts ERA5 variables to the format expected by the TC track detection
    algorithm, including:
    - Renaming variables to match EWB conventions
    - Converting units (Pa to hPa if needed)
    - Ensuring proper coordinate structure
    
    Args:
        era5_ds: Filtered ERA5 dataset from filter_era5_for_tc.py
        
    Returns:
        Tuple of (sea_level_pressure, wind_speed, geopotential_thickness) DataArrays
    """
    # Check required variables
    required_vars = [
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential_300",
        "geopotential_500",
    ]

    print("  Checking for required variables...")

    missing = [var for var in required_vars if var not in era5_ds.feature.values]
    if missing:
        raise ValueError(
            f"Missing required variables: {missing}. "
            f"Available: {list(era5_ds.fearures.values)}"
        )

    # print("  Creating restructured xarray handle...")

    # Start with coordinates and dimensions from original dataset
    modified_ds = xr.Dataset(
        coords={k: v for k, v in era5_ds.coords.items() if k != "feature"}
    )
    
    
    # print(modified_ds.init_time.values)

    
    # Convert feature dimension to data variables
    # This preserves lazy loading by using xarray's indexing without .load() or .compute()
    # print("  Converting 'feature' dimension to data variables...")
    
    # Get the name of the data variable that contains the feature data
    # Typically there's one main data variable indexed by the feature dimension
    data_var_name = list(era5_ds.data_vars)[0]
    
    for feature_name in era5_ds.feature.values:
        # Select data for this feature and assign as a named variable
        # Using .sel() maintains lazy evaluation - data is not loaded until explicitly computed
        # Extract the DataArray from the Dataset result by accessing the underlying variable
        feature_data = era5_ds[data_var_name].sel(feature=feature_name)
        modified_ds[feature_name] = feature_data

    modified_ds = modified_ds.reset_coords("feature", drop=True)
    modified_ds = modified_ds[required_vars]

    # print("  Convert lead_time dimension to different notational convention...")

    # print(f"before lead time values: {modified_ds.lead_time.values}")
    

    # Convert lead_time from int32 hours to timedelta64[ns]
    lead_time_hours = modified_ds.coords["lead_time"].values
    # First convert to timedelta64[h] (hours), then to ns
    modified_ds = modified_ds.assign_coords(
        lead_time=lead_time_hours.astype("timedelta64[h]").astype("timedelta64[ns]")
    )

    # print(f"after lead time values: {modified_ds.lead_time.values}")

    if event_start is None:
        earliest_prediction = min(modified_ds.coords["time"].values)
    else:
        earliest_prediction = event_start
        # print(type(event_start), event_start)
        # print(min(modified_ds.coords["time"].values))
        
    # print(earliest_prediction)

    init_time_vals = earliest_prediction - modified_ds.coords["lead_time"].values
    modified_ds = modified_ds.assign_coords(
        lead_time=init_time_vals
    ).rename({"lead_time": "init_time"}) # .isel(init_time=[-1])
    modified_ds = modified_ds.dropna(dim="time", how="all")

    # print(f"init time values: {modified_ds.init_time.values}")

    # Combine geopotential_300 and geopotential_500 into a single geopotential variable with level coordinate
    # This preserves lazy loading by using xr.concat which doesn't trigger computation
    # print("  Combining geopotential_300 and geopotential_500 into geopotential with level coordinate...")
    geopotential_levels = xr.concat(
        [modified_ds["geopotential_300"], modified_ds["geopotential_500"]],
        dim="level"
    )
    geopotential_levels = geopotential_levels.assign_coords(level=[300, 500])
    modified_ds["geopotential"] = geopotential_levels
    
    # Remove the individual level-specific variables
    modified_ds = modified_ds.drop_vars(["geopotential_300", "geopotential_500"])
    # print("  Created geopotential variable with level coordinate [300, 500]")
    
    # print(f"  Converted {len(era5_ds.feature.values)} feature to data variables")
    
    # Extract and prepare sea level pressure
    # Ensure it's in Pa (ERA5 typically uses Pa)
    # slp = era5_ds["mean_sea_level_pressure"]
    
    # print("  Renaming variables...")

    modified_ds = modified_ds.rename(
        {
            "time": "valid_time",
            "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
            "10m_u_component_of_wind": "surface_eastward_wind",
            "10m_v_component_of_wind": "surface_northward_wind"
        }
    )

    # print(modified_ds)

    # modified_ds = modified_ds.rename_dims({"time": "init_time"})
    
    # Calculate wind speed from u and v components
    # u_wind = era5_ds["10m_u_component_of_wind"]
    # v_wind = era5_ds["10m_v_component_of_wind"]
    # wind_speed = calc.calculate_wind_speed(u_wind, v_wind)
    
    # Calculate geopotential thickness (500hPa - 300hPa geopotential difference)
    # ERA5 uses levels, so we need to extract 500hPa and 300hPa
    
    # print("  Calculating geopotential thickness...")

    geopotential_thickness = calc.geopotential_thickness(modified_ds["geopotential"], geopotential=True)
    modified_ds["geopotential_thickness"] = geopotential_thickness

    modified_ds = calc.maybe_calculate_wind_speed(modified_ds)
    

    # valid_time = modified_ds.init_time.values + modified_ds.lead_time.values
    # modified_ds = modified_ds.assign_coords(valid_time=("lead_time", valid_time))
    
    if il_output:
        # print("  Converting to init_time coordinates...")
        modified_ds = utils.convert_init_and_valid_time_to_lead_and_init_time(modified_ds)
        return modified_ds
    else:
        # print("  Converting to valid_time coordinates...")
        modified_ds = utils.convert_init_and_valid_time_to_lead_time(modified_ds)
        # modified_ds = modified_ds.assign_coords(init_time=modified_ds.init_time.values.ravel()[0])
        
        return modified_ds


def extract_ibtracs_path_data_natively(case_ids: list[int]) -> dict[int, pd.DataFrame]:
    """
    Extracts tropical cyclone track data directly from EWB's native IBTrACS class.
    """
    cases_list = cases.load_ewb_events_yaml_into_case_list()
    results = {}
    
    ibtracs_target = ewb.inputs.IBTrACS()
    lazy_df = ibtracs_target._open_data_from_source()
    
    for case_id in case_ids:
        case_obj = next((c for c in cases_list if c.case_id_number == case_id), None)
        if not case_obj:
            print(f"Case ID {case_id} not found in EWB event list.")
            continue
            
        print(f"[INFO] Extracting IBTrACS track for Case ID {case_id} ({getattr(case_obj, 'title', 'Unknown')})")
        
        df_all = lazy_df.collect().to_pandas()

        
        if hasattr(case_obj, 'title') and case_obj.title:
            storm_names = [name.upper() for name in utils.extract_tc_names(case_obj.title)]
            filtered_df = df_all[df_all['NAME'].str.upper().isin(storm_names)]

        else:
            start_date = pd.Timestamp(case_obj.start_date)
            df_all['ISO_TIME'] = pd.to_datetime(df_all['ISO_TIME'])
            filtered_df = df_all[
                (df_all['ISO_TIME'] >= start_date - pd.Timedelta(days=5)) & 
                (df_all['ISO_TIME'] <= start_date + pd.Timedelta(days=10))
            ]
            
        if not filtered_df.empty:
            # --- FALLBACK LOGIC FOR MISSING INTENSITY DATA ---
            if 'WMO_PRES' in filtered_df.columns and 'USA_PRES' in filtered_df.columns:
                filtered_df['WMO_PRES'] = filtered_df['WMO_PRES'].fillna(filtered_df['USA_PRES'])
            if 'WMO_WIND' in filtered_df.columns and 'USA_WIND' in filtered_df.columns:
                filtered_df['WMO_WIND'] = filtered_df['WMO_WIND'].fillna(filtered_df['USA_WIND'])
            # -------------------------------------------------

            rename_map = {
                'ISO_TIME': 'valid_time',
                'LAT': 'latitude',
                'LON': 'longitude',
                'WMO_PRES': 'air_pressure_at_mean_sea_level',
                'WMO_WIND': 'surface_wind_speed'
            }
            filtered_df = filtered_df.rename(columns={k: v for k, v in rename_map.items() if k in filtered_df.columns})
            
            if 'air_pressure_at_mean_sea_level' in filtered_df.columns:
                filtered_df['air_pressure_at_mean_sea_level'] = pd.to_numeric(
                    filtered_df['air_pressure_at_mean_sea_level'], errors='coerce'
                ) * 100.0
                
            if 'surface_wind_speed' in filtered_df.columns:
                filtered_df['surface_wind_speed'] = pd.to_numeric(
                    filtered_df['surface_wind_speed'], errors='coerce'
                )
                
            if 'valid_time' in filtered_df.columns:
                filtered_df['valid_time'] = pd.to_datetime(filtered_df['valid_time']).dt.tz_localize(None)
                filtered_df = filtered_df.drop_duplicates(subset=['valid_time'])
                filtered_df = filtered_df.set_index('valid_time')
                
            results[case_id] = filtered_df
        else:
            print(f"No matching IBTrACS records found for Case ID {case_id}")
            
    return results

