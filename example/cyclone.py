"""
Script to extract tropical cyclone tracks from filtered ERA5 zarr files
and compare them directly against IBTrACS data using TempestExtremes,
bypassing grid-based runners completely.
"""

import os
import argparse
from pathlib import Path
import pathlib
import re

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import extremeweatherbench as ewb

from extremeweatherbench import calc, utils, cases
from extremeweatherbench.derived import TropicalCycloneTrackVariables
from cleandata import prepare_era5_data_for_tc_detection, extract_ibtracs_path_data_natively


def reformat_track_ds(track_ds, ensemble_member, init_time_index):
    new_ds = utils.convert_valid_time_to_init_time(track_ds)
    new_ds = new_ds.reset_coords(["latitude", "longitude"], drop=False)
    new_ds = new_ds.expand_dims({"ensemble": 1}).assign_coords(ensemble=[ensemble_member])
    new_ds = new_ds.expand_dims({"init_index": 1}).assign_coords(init_index=[init_time_index])
    return new_ds


def get_structured_group_list(zarr_path: str, event_types: list[str] = ["heat_wave", "freeze", "tropical_cyclone"]) -> list[tuple[str, int, int, int]]:
    zarr_handle = zarr.open(zarr_path)
    raw_group_list = list(zarr_handle.group_keys())
    
   
    if not raw_group_list or not any(g.startswith("events_") for g in raw_group_list):
        path_str = str(zarr_path)
        match = re.search(r'events_tropical_cyclone_(\d+)_lt(m?\d+)_off(\d+)', path_str)
        if match:
            case_id = int(match.group(1))
            lead_str = match.group(2)
            offset = int(match.group(3))
            lead_time = -int(lead_str[1:]) if lead_str.startswith('m') else int(lead_str)
            return [("tropical_cyclone", case_id, offset, lead_time)]
        
        match_simple = re.search(r'events_tropical_cyclone_(\d+)', path_str)
        if match_simple:
            return [("tropical_cyclone", int(match_simple.group(1)), 0, 0)]

    filtered_group_list = [entry[7:] for entry in raw_group_list if entry.startswith("events_")]

    pat = re.compile(r'^(?P<id>\d+)_lt(?P<lead>m?\d+)_off(?P<offset>\d+)$')

    structured_group_list = []
    for entry in filtered_group_list:
        for event_type in event_types:
            if entry.startswith(event_type):
                found_type = event_type
                m = pat.fullmatch(entry[len(event_type) + 1:])
                if m:
                    match_dict = m.groupdict()
                    case_id = int(match_dict["id"])
                    offset = int(match_dict["offset"])
                    if "m" in match_dict["lead"]:
                        lead_time = -int(match_dict["lead"][1:])
                    else:
                        lead_time = int(match_dict["lead"])

                    structured_group_list.append((found_type, case_id, offset, lead_time))
                break
            
    return structured_group_list


def get_case_init_time(case_id):
    cases_list = cases.load_ewb_events_yaml_into_case_list()
    for case in cases_list:
        if case.case_id_number == case_id:
            return pd.Timestamp(case.start_date).to_datetime64().astype("datetime64[ns]")
    return None


def extract_tc_track_from_zarr(
    zarr_path: str,
    output_csv_path: str = ".",
    event_types: list[str] = ["tropical_cyclone"],
    num_ensembles: int = None,
    **track_detection_kwargs,
) -> pd.DataFrame:
    """Extract tropical cyclone track from filtered ERA5 zarr file."""
    variable_calculator = TropicalCycloneTrackVariables(**track_detection_kwargs)

    print(f"Loading zarr path: {zarr_path}")
    structured_group_list = get_structured_group_list(zarr_path, event_types=event_types)
    print(f"Detected sub-groups/cases to process: {structured_group_list}")

    lt_ds_list = []
    event_id = None
    ibtracs_results = {}

    for entry in structured_group_list:
        event_type, case_id, offset, lead_time = entry
        event_id = case_id
        print(f"\nEvent {case_id}, Type: {event_type}")

        match event_type:
            case "tropical_cyclone":
                try:
                    group_name = f"events_{event_type}_{case_id}_lt{lead_time if lead_time >= 0 else f'm{-lead_time}'}_off{offset}"
                    era5_ds = xr.open_dataset(zarr_path, engine="zarr", group=group_name)
                except Exception:
                    era5_ds = xr.open_dataset(zarr_path, engine="zarr", consolidated=False)
                
                ensemble_members = era5_ds.ensemble.values if "ensemble" in era5_ds.coords else [0]
                
                # --- APPLY ENSEMBLE LIMIT ---
                if num_ensembles is not None:
                    ensemble_members = ensemble_members[:num_ensembles]
                    # print(f"Limiting to first {num_ensembles} ensemble members: {ensemble_members}")
                # ----------------------------

                # print("=" * 70)
                # print("Extracting IBTrACS Path Data for Tropical Cyclone Events")
                # print("=" * 70)
                
                ibtracs_results = extract_ibtracs_path_data_natively([case_id])
        
                for ib_event_id, df in ibtracs_results.items():
                    ds = xr.Dataset.from_dataframe(df)
                    lt = lead_time
                    # print(era5_ds)
                    # print(era5_ds.time.values)
                    # print(era5_ds.lead_time.values.astype("timedelta64[h]").astype(int))
                   
                    print(f"\nPreparing data with lt {lt} for TC track detection...")
                    reformatted_ds = prepare_era5_data_for_tc_detection(era5_ds)
                    # print(reformatted_ds)
                    # print(reformatted_ds.valid_time.values)
                    # print(reformatted_ds.lead_time.values.astype("timedelta64[h]").astype(int))
                    # print(reformatted_ds.init_time.values)

                    member_ds_list = []

                    for member in ensemble_members:
                        member_ds = reformatted_ds.sel(ensemble=member) if "ensemble" in reformatted_ds.dims else reformatted_ds
                        # print(f"Ensemble member {member}")
                        # print("membseds",member_ds["surface_wind_speed"].values.flat[0])
                        # print("*"*100)
                        # print(ds["valid_time"])
                        
                        new_track_ds = variable_calculator.get_or_compute_tracks(data=member_ds, _target_dataset=ds)
                        # ------------------------------------------------------------------------------------------
                        
                        if "lead_time" in new_track_ds.dims and len(new_track_ds.lead_time.values) < 1:
                            print(f"No viable track found for ensemble member {member}")
                            continue

                        track_ds = reformat_track_ds(new_track_ds, member, lt)
                      
                        member_ds_list.append(track_ds)

                    if member_ds_list:
                       
                        lt_ds_list.append(xr.concat(member_ds_list, dim="ensemble", join="outer", coords="different",compat="equals"))

            case _:
                print(f"Unimplemented event type {event_type}")
                continue
    
    if not lt_ds_list:
        print("No valid tracks extracted across groups.")
        return pd.DataFrame()

    final_ds = xr.concat(lt_ds_list, dim="init_index", join="outer", coords="different",compat="equals")
    
    # --- MERGE ACTUAL DATA AND CALCULATE METRICS ---
    final_df = final_ds.to_dataframe().dropna().reset_index()
    
    if event_id in ibtracs_results:
        target_df = ibtracs_results[event_id].reset_index()
        print(f"Total IBTrACS target points for event {event_id}: {len(target_df)}")
        
        # Agar aapko specifically valid_time ki count dekhni hai
        if 'valid_time' in target_df.columns:
            print(f"Total valid time steps: {target_df['valid_time'].nunique()}")
        target_df = target_df[['valid_time', 'latitude', 'longitude', 'air_pressure_at_mean_sea_level', 'surface_wind_speed']].rename(
            columns={
                'latitude': 'actual_latitude',
                'longitude': 'actual_longitude',
                'air_pressure_at_mean_sea_level': 'actual_pressure',
                'surface_wind_speed': 'actual_wind_speed'
            }
        )
        # Inner merge aligns predicted and actual by valid_time
        merged_df = pd.merge(final_df, target_df, on='valid_time', how='inner')
        
    else:
        merged_df = final_df
        print("Could not merge actual IBTrACS data. Check if Target data exists.")
    # -----------------------------------------------
    # 1. Get the exact directory where this Python script is located
    script_dir = pathlib.Path(__file__).resolve().parent

    # 2. Combine the script directory with your desired output path
    # (If output_csv_path is an absolute path, pathlib handles it smartly)
    output_dir = script_dir / output_csv_path

    # 3. Define the full filepath (combining the dir, the "csv" folder, and the filename)
    filepath = output_dir / "example/csv" / f"tc_track_event_{event_id}_tracks.csv"

    # 4. Create all required folders automatically (including the new "csv" subfolder)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 5. Save the file and print
    merged_df.to_csv(filepath, index=False)
    print(f"Saved merged dataset with actuals and errors: {filepath}")
    return merged_df


def main():
    target_case_id = 255
    # default_zarr_path = "/p/scratch/hai_1247/beast_forecast/cyclone_e10.zarr/events_tropical_cyclone_256_ltm36_off0"
    default_zarr_path = "/p/scratch/hai_1247/beast_forecast/forecasts-1950741.zarr/events_tropical_cyclone_255_lt0_off0"

    parser = argparse.ArgumentParser(
        description="Extract tropical cyclone track and compare directly against IBTrACS"
    )
    parser.add_argument(
        "zarr_path",
        type=str,
        nargs="?",
        default=default_zarr_path,
        help="Path to the filtered ERA5 zarr file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.getcwd(), 
        help="Output CSV file path directory (default: current working directory)"
    )
    # --- ADDED NUM ENSEMBLES ARGUMENT ---
    parser.add_argument(
        "--num-ensembles",
        type=int,
        default=100,
        help="Number of ensemble members to process (default: all)"
    )
    parser.add_argument(
        "--slp-contour",
        type=float,
        default=200.0,
        help="SLP contour threshold in Pa (default: 200.0)"
    )
    parser.add_argument(
        "--min-wind-timesteps",
        type=int,
        default=3,
        help="Minimum timesteps with wind >= 10 m/s"
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=50.0,
        help="Maximum latitude for TC detection"
    )

    
    args = parser.parse_args()
    
    # if args.output is None:
    #     args.output = str(Path(args.zarr_path).parent)

    extract_tc_track_from_zarr(
        args.zarr_path,
        args.output,
        num_ensembles=args.num_ensembles,
        slp_contour_magnitude=args.slp_contour,
        timestep_count_wind_minimum=args.min_wind_timesteps,
        latitude_max_degrees=args.max_lat,
    )


if __name__ == "__main__":
    main()