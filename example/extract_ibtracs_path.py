"""
Script to extract latitude and longitude information over time from IBTrACS data,
filtered by tropical cyclone event IDs listed in extremeweatherbench/data/events.yaml.
"""

import polars as pl
import pandas as pd
import xarray as xr
from extremeweatherbench import cases, inputs
from extremeweatherbench.defaults import ibtracs_target


def get_tropical_cyclone_event_ids():
    """Load all EWB events and filter for tropical cyclone case IDs.
    
    Returns:
        list: List of case_id_numbers for tropical cyclone events.
    """
    cases_list = cases.load_ewb_events_yaml_into_case_list()
    tc_event_ids = [
        c.case_id_number 
        for c in cases_list 
        if c.event_type == "tropical_cyclone"
    ]
    return tc_event_ids


def get_tc_names_for_event_ids(event_ids):
    """Extract tropical cyclone names for given event IDs.
    
    Args:
        event_ids: List of case_id_numbers to get names for.
        
    Returns:
        dict: Dictionary mapping event_id to list of possible TC names.
    """
    cases_list = cases.load_ewb_events_yaml_into_case_list()
    tc_names_map = {}
    
    for case in cases_list:
        if case.case_id_number in event_ids and case.event_type == "tropical_cyclone":
            # Use the same utility function that IBTrACS uses internally
            from extremeweatherbench import utils
            possible_names = utils.extract_tc_names(case.title)
            tc_names_map[case.case_id_number] = possible_names
    
    return tc_names_map


def extract_ibtracs_path_data(event_ids=None, verbose=False):
    """Extract latitude and longitude information from IBTrACS data.
    
    Args:
        event_ids: Optional list of specific case_id_numbers to filter.
                   If None, extracts all tropical cyclone events.
    
    Returns:
        dict: Dictionary with event_id as key and DataFrame containing
              valid_time, latitude, longitude, tc_name, season, and number.
    """
    # Get tropical cyclone event IDs if not provided
    if event_ids is None:
        event_ids = get_tropical_cyclone_event_ids()
    
    # Get TC names for each event ID
    tc_names_map = get_tc_names_for_event_ids(event_ids)
    
    # Load IBTrACS data using polars (same as IBTrACS target does)
    if verbose:
        print("Loading IBTrACS data...")
    # ibtracs_data = pl.scan_csv(
    #     ibtracs_target.source,
    #     storage_options=ibtracs_target.storage_options,
    #     skip_rows_after_header=1,
    # )

    ibtracs_data = ibtracs_target._open_data_from_source()
    
    # Extract unique seasons from the event IDs
    cases_list = cases.load_ewb_events_yaml_into_case_list()
    seasons = set()
    for case in cases_list:
        if case.case_id_number in event_ids:
            season = case.start_date.year
            if case.start_date.month > 11:
                season += 1
            seasons.add(season)

    if verbose:
        print(f"Filtering for seasons: {sorted(seasons)}")
        print(f"Filtering for event IDs: {event_ids}")
    
    # Collect all TC names we're looking for
    all_tc_names = []
    for names in tc_names_map.values():
        all_tc_names.extend(names)

    if verbose:
        print(f"Filtering for TC names: {all_tc_names}")
    
    # Filter IBTrACS data by season and TC name (using original IBTrACS column names)
    # IBTrACS uses uppercase: SEASON, NAME, LAT, LON, ISO_TIME, NUMBER, USA_WIND, USA_PRES
    filtered_data = ibtracs_data.filter(
        (pl.col("SEASON").cast(pl.Int64).is_in(list(seasons))) &
        pl.col("NAME").is_in(all_tc_names)
    )
    
    # Select only the columns we need (original IBTrACS names)
    columns_to_keep = [
        "ISO_TIME",
        "NAME",
        "SEASON",
        "NUMBER",
        "LAT",
        "LON",
        "USA_WIND",
        "USA_PRES",
    ]
    
    filtered_data = filtered_data.select(columns_to_keep)
    
    # Filter out rows with null wind speed or pressure
    filtered_data = filtered_data.filter(
        pl.col("USA_WIND").is_not_null() &
        pl.col("USA_PRES").is_not_null()
    )
    
    # Collect to pandas DataFrame
    if verbose:
        print("Collecting filtered data...")

    df = filtered_data.collect(engine="streaming").to_pandas()
    
    # Rename columns from IBTrACS names to EWB standard names
    df = df.rename(columns={
        "ISO_TIME": "valid_time",
        "NAME": "tc_name",
        "SEASON": "season",
        "NUMBER": "number",
        "LAT": "latitude",
        "LON": "longitude",
        "USA_WIND": "surface_wind_speed",
        "USA_PRES": "air_pressure_at_mean_sea_level",
    })
    
    # Convert valid_time to datetime
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    
    # Sort by valid_time for each storm
    # df = df.sort_values(["tc_name", "valid_time"])
    
    # Organize results by event_id
    results = {}
    for event_id in event_ids:
        if verbose:
            print(f"  Searching for event id {event_id}")

        case = next(
            (c for c in cases_list if c.case_id_number == event_id), 
            None
        )
        if case:
            tc_names = tc_names_map.get(event_id, [])
            # Filter dataframe for this specific event's TC names
            event_df = df[df["tc_name"].isin(tc_names)].copy()
            if len(event_df) > 0:
                event_df = event_df.sort_values(["tc_name", "valid_time"])
                results[event_id] = event_df
                if verbose:
                    print(f"Event {event_id} ({case.title}): {len(event_df)} track points")
            else:
                if verbose:
                    print(f"Event {event_id} ({case.title}): No matching IBTrACS data found")
    
    return results


def save_paths_to_files(results, output_dir="./ibtracs_paths"):
    """Save extracted path data to CSV files.
    
    Args:
        results: Dictionary of DataFrames keyed by event_id.
        output_dir: Directory to save output files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for event_id, df in results.items():
        filename = f"event_{event_id}_path.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    # Example usage: Extract paths for all tropical cyclones
    print("=" * 70)
    print("Extracting IBTrACS Path Data for Tropical Cyclone Events")
    print("=" * 70)
    
    # Get all tropical cyclone event IDs
    tc_event_ids = [256, 167, 192] # get_tropical_cyclone_event_ids()
    print(f"\nFound {len(tc_event_ids)} tropical cyclone events")
    
    # Extract the path data
    results = extract_ibtracs_path_data(tc_event_ids)
    
    # Save to CSV files
    save_paths_to_files(results)
    
    # # Example: Access data for a specific event
    # print("\n" + "=" * 70)
    # print("Example: Displaying first few rows for Event 151 (Milton)")
    # print("=" * 70)
    # if 151 in results:
    #     print(results[151][["valid_time", "latitude", "longitude", "tc_name"]].head())
    # else:
    #     print("Event 151 not found in results")