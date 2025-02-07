"""Utility functions for the ExtremeWeatherBench package that don't fit into any
other specialized package.
"""

from typing import Union, List, Literal, Optional
from collections import namedtuple
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import ujson
import rioxarray  # noqa: F401
import xarray as xr
from kerchunk.hdf import SingleHdf5ToZarr
from shapely.geometry import box
import datetime
from pathlib import Path
from importlib import resources
import yaml
import pickle

#: Struct packaging latitude/longitude location definitions.
Location = namedtuple("Location", ["latitude", "longitude"])

#: Maps the ARCO ERA5 to CF conventions.
ERA5_MAPPING = {
    "air_temperature": "2m_temperature",
    "eastward_wind": "10m_u_component_of_wind",
    "northward_wind": "10m_v_component_of_wind",
    "air_pressure_at_mean_sea_level": "mean_sea_level_pressure",
    "specific_humidity": "specific_humidity",
    "valid_time": "time",
    "level": "level",
    "latitude": "latitude",
    "longitude": "longitude",
}


def convert_longitude_to_360(longitude: float) -> float:
    """Convert a longitude from the range [-180, 180) to [0, 360)."""
    return np.mod(longitude, 360)


def convert_longitude_to_180(
    dataset: Union[xr.Dataset, xr.DataArray], longitude_name: str = "longitude"
) -> Union[xr.Dataset, xr.DataArray]:
    """Coerce the longitude dimension of an xarray data structure to [-180, 180)."""
    dataset.coords[longitude_name] = (dataset.coords[longitude_name] + 180) % 360 - 180
    dataset = dataset.sortby(longitude_name)
    return dataset


def generate_json_from_nc(u, so, fs, fs_out, json_dir):
    """Generate a kerchunk JSON file from a NetCDF file."""
    with fs.open(u, **so) as infile:
        h5chunks = SingleHdf5ToZarr(infile, u, inline_threshold=300)

        file_split = u.split(
            "/"
        )  # seperate file path to create a unique name for each json
        model = file_split[1].split("_")[0]
        date_string = file_split[-1].split("_")[3]
        outf = f"{json_dir}{model}_{date_string}_.json"
        print(outf)
        with fs_out.open(outf, "wb") as f:
            f.write(ujson.dumps(h5chunks.translate()).encode())


def clip_dataset_to_bounding_box(
    dataset: xr.Dataset,
    location_center: Location,
    length_km: float,
) -> xr.Dataset:
    """Clip an xarray dataset to a boxbox around a given location.

    Args:
        dataset: The input xarray dataset.
        location_center: A Location object corresponding to the center of the bounding box.
        length_km: The side length of the bounding box in kilometers.

    Returns:
        The clipped xarray dataset.
    """
    lat_center = location_center.latitude
    lon_center = location_center.longitude
    if lon_center < 0:
        lon_center = convert_longitude_to_360(lon_center)
    # Convert length from kilometers to degrees (approximation)
    length_deg = length_km / 111  # 1 degree is approximately 111 km

    # Create a bounding box
    min_lat = lat_center - (length_deg / 2)
    max_lat = lat_center + (length_deg / 2)
    min_lon = lon_center - (length_deg / 2)
    max_lon = lon_center + (length_deg / 2)

    # Create a GeoDataFrame with the bounding box
    bbox = gpd.GeoDataFrame(
        {"geometry": [box(min_lon, min_lat, max_lon, max_lat)]},
    )
    # Clip the dataset using the bounding box
    clipped_dataset = dataset.rio.write_crs("EPSG:4326").rio.clip(
        bbox.geometry, bbox.crs, drop=True
    )

    return clipped_dataset


def convert_day_yearofday_to_time(dataset: xr.Dataset, year: int) -> xr.Dataset:
    """Convert dayofyear and hour coordinates in an xarray Dataset to a new time
    coordinate.

    Args:
        dataset: The input xarray dataset.
        year: The base year to use for the time coordinate.

    Returns:
        The dataset with a new time coordinate.
    """
    # Create a new time coordinate by combining dayofyear and hour
    time_dim = pd.date_range(
        start=f"{year}-01-01",
        periods=len(dataset["dayofyear"]) * len(dataset["hour"]),
        freq="6h",
    )
    dataset = dataset.stack(time=("dayofyear", "hour")).drop(["dayofyear", "hour"])
    # Assign the new time coordinate to the dataset
    dataset = dataset.assign_coords(time=time_dim)

    return dataset


def remove_ocean_gridpoints(dataset: xr.Dataset) -> xr.Dataset:
    """Subset a dataset to only include land gridpoints based on a land-sea mask.

    Args:
        dataset: The input xarray dataset.

    Returns:
        The dataset masked to only land gridpoints.
    """
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_sea_mask = land.mask(dataset.longitude, dataset.latitude)
    land_mask = land_sea_mask == 0
    # Subset the dataset to only include land gridpoints
    dataset = dataset.where(land_mask, drop=True)

    return dataset


def _open_kerchunk_zarr_reference_jsons(
    file_list, forecast_schema_config, remote_protocol: str = "s3"
):
    """Open a dataset from a list of kerchunk JSON files."""
    xarray_datasets = []
    for json_file in file_list:
        fs_ = fsspec.filesystem(
            "reference",
            fo=json_file,
            ref_storage_args={"skip_instance_cache": True},
            remote_protocol=remote_protocol,
            remote_options={"anon": True},
        )
        m = fs_.get_mapper("")
        ds = xr.open_dataset(
            m, engine="zarr", backend_kwargs={"consolidated": False}, chunks="auto"
        )
        if "initialization_time" not in ds.attrs:
            raise ValueError(
                "Initialization time not found in dataset attributes. \
                             Please add initialization_time to the dataset attributes."
            )
        else:
            model_run_time = np.datetime64(
                pd.to_datetime(ds.attrs["initialization_time"])
            )
        ds[forecast_schema_config.init_time] = model_run_time.astype("datetime64[ns]")
        ds = ds.set_coords(forecast_schema_config.init_time)
        ds = ds.expand_dims(forecast_schema_config.init_time)
        for variable in forecast_schema_config.__dict__:
            attr_value = getattr(forecast_schema_config, variable)
            if attr_value in ds.data_vars:
                ds = ds.rename({attr_value: variable})
        ds = ds.transpose(
            forecast_schema_config.init_time,
            forecast_schema_config.valid_time,
            forecast_schema_config.latitude,
            forecast_schema_config.longitude,
            forecast_schema_config.level,
        )
        xarray_datasets.append(ds)

    return xr.concat(xarray_datasets, dim=forecast_schema_config.init_time)


def _open_mlwp_kerchunk_reference(
    file, forecast_schema_config, remote_protocol: str = "s3"
):
    """Open a dataset from a kerchunked reference file for the OAR MLWP S3 bucket."""
    if "parq" in file:
        storage_options = {
            "remote_protocol": "s3",
            "skip_instance_cache": True,
            "remote_options": {"anon": True},
            "target_protocol": "file",
            "lazy": True,
        }  # options passed to fsspec
        open_dataset_options: dict = {"chunks": {}}  # opens passed to xarray

        ds = xr.open_dataset(
            file,
            engine="kerchunk",
            storage_options=storage_options,
            open_dataset_options=open_dataset_options,
        )
    else:
        ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": file,
                    "remote_protocol": remote_protocol,
                    "remote_options": {"anon": True},
                },
            },
        )
    ds = ds.rename({"time": "lead_time"})
    ds["lead_time"] = range(0, 241, 6)
    for variable in forecast_schema_config.__dict__:
        attr_value = getattr(forecast_schema_config, variable)
        if attr_value in ds.data_vars:
            ds = ds.rename({attr_value: variable})
        # Step 1: Create a meshgrid of init_time and lead_time
    lead_time_grid, init_time_grid = np.meshgrid(ds.lead_time, ds.init_time)
    # Step 2: Flatten the meshgrid and convert lead_time to timedelta
    valid_time = init_time_grid.flatten() + pd.to_timedelta(
        lead_time_grid.flatten(), unit="h"
    )
    ds.coords["time"] = valid_time
    return ds


def map_era5_vars_to_forecast(forecast_schema_config, forecast_dataset, era5_dataset):
    """Map ERA5 variable names to forecast variable names."""
    era5_subset_list = []
    for variable in forecast_schema_config.__dict__:
        if variable in forecast_dataset.data_vars:
            era5_dataset = era5_dataset.rename({ERA5_MAPPING[variable]: variable})
            era5_subset_list.append(variable)
    return era5_dataset[era5_subset_list]


def expand_lead_times_to_6_hourly(
    dataarray: xr.DataArray, max_fcst_hour: int = 240, fcst_output_cadence: int = 6
) -> xr.DataArray:
    """Makes hours in metrics output for max_fcst_hour hours at a fcst_cadence-hourly rate.
    Depending on initialization time and output cadence, there may be missing lead times
    in final output of certain metrics."""
    all_hours = np.arange(0, max_fcst_hour + 1, fcst_output_cadence)
    final_data = []
    final_times = []
    current_idx = 0
    for hour in all_hours:
        if (
            current_idx < len(dataarray.lead_time)
            and dataarray.lead_time[current_idx] == hour
        ):
            final_data.append(dataarray.values[current_idx])
            current_idx += 1
        else:
            final_data.append(None)
        final_times.append(hour)
    dataarray = xr.DataArray(
        data=final_data, dims=["lead_time"], coords={"lead_time": final_times}
    ).astype(float)
    return dataarray


def process_dataarray_for_output(da_list: List[xr.DataArray]) -> xr.DataArray:
    """Extract and format data from a list of DataArrays.

    Args:
        da_list: A list of xarray DataArrays.

    Returns:
        A DataArray with a sorted lead_time coordinate, expanded to 6-hourly intervals.
    """

    if len(da_list) == 0:
        # create dummy nan dataarray in case no applicable forecast valid times exist
        output_da = xr.DataArray(
            data=[np.nan],
            dims=["lead_time"],
            coords={"lead_time": [0]},
        )
    else:
        output_da = xr.concat(da_list, dim="lead_time")
        # Reverse the lead time so that the minimum lead time is first
        output_da = output_da.isel(lead_time=slice(None, None, -1))
    output_da = expand_lead_times_to_6_hourly(output_da)
    return output_da


def center_forecast_on_time(da: xr.DataArray, time: pd.Timestamp, hours: int):
    """Center a forecast DataArray on a given time with a given range in hours.

    Args:
        da: The forecast DataArray to center.
        time: The time to center the forecast on.
        hours: The number of hours to include in the centered forecast.
    """
    time_range = pd.date_range(
        end=pd.to_datetime(time) + pd.Timedelta(hours=hours),
        periods=hours * 2 + 1,
        freq="h",
    )
    return da.sel(time=slice(time_range[0], time_range[-1]))


def temporal_align_dataarrays(
    forecast: xr.DataArray,
    observation: xr.DataArray,
    init_time_datetime: datetime.datetime,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align the individual initialization time forecast and observation dataarrays.

    Args:
        forecast: The forecast dataarray to align.
        observation: The observation dataarray to align.
        init_time_datetime: The initialization time to subset the forecast dataarray by.

    Returns:
        A tuple containing the time-aligned forecast and observation dataarrays.
    """

    forecast = forecast.sel(init_time=init_time_datetime)
    time = np.array(
        [init_time_datetime + pd.Timedelta(hours=int(t)) for t in forecast["lead_time"]]
    )
    forecast = forecast.assign_coords(time=("lead_time", time))
    forecast = forecast.swap_dims({"lead_time": "time"})
    forecast, observation = xr.align(forecast, observation, join="inner")
    return (forecast, observation)


def align_observations_temporal_resolution(
    forecast: xr.DataArray, observation: xr.DataArray
) -> xr.DataArray:
    """Align observation dataarray on the forecast dataarray's temporal resolution.,

    Metrics which need a singular timestep from gridded obs will fail if the forecasts
    are not aligned with the observation timestamps (e.g. a 03z minimum temp in observations
    when the forecast only has 00z and 06z timesteps).

    Args:
        forecast: The forecast data which will be aligned against.
        observation: The observation data to align.

    Returns:
        The aligned observation dataarray.
    """
    obs_time_delta = pd.to_timedelta(np.diff(observation.time).mean())
    forecast_time_delta = pd.to_timedelta(np.diff(forecast.lead_time).mean(), unit="h")

    if forecast_time_delta > obs_time_delta:
        observation = observation.resample(time=forecast_time_delta).first()

    return observation


def truncate_incomplete_days(da: xr.DataArray) -> xr.DataArray:
    """Truncate a dataarray to only include full days of data."""
    # Group by dayofyear and check if each day has a complete times
    # Count how many unique hours exist per day in the data
    hours_per_day = len(np.unique(da.time.dt.hour.values))
    valid_days = da.groupby("time.dayofyear").count("time") == hours_per_day
    # Only keep days that have a full set of timestamps
    da = da.where(
        da.time.dt.dayofyear.isin(
            valid_days.where(valid_days).dropna(dim="dayofyear").dayofyear
        ),
        drop=True,
    )
    return da


def return_max_min_timestamp(da: xr.DataArray) -> pd.Timestamp:
    """Return the timestamp of the maximum minimum temperature in a DataArray."""
    return pd.Timestamp(
        da.where(
            da == da.groupby("time.dayofyear").min().max(),
            drop=True,
        ).time.values[0]
    )


def load_events_yaml():
    """Load the events yaml file."""
    import extremeweatherbench.data

    events_yaml_file = resources.files(extremeweatherbench.data).joinpath("events.yaml")
    with resources.as_file(events_yaml_file) as file:
        yaml_event_case = read_event_yaml(file)

    return yaml_event_case


def read_event_yaml(input_pth: str | Path) -> dict:
    """Read events yaml from data."""
    input_pth = Path(input_pth)
    with open(input_pth, "rb") as f:
        yaml_event_case = yaml.safe_load(f)
    return yaml_event_case


# BallTree algorithm from Herbie
# (https://github.com/blaylockbk/Herbie/blob/a96cbf5ea864c9a85975bf98fa90a56381c9aa75/herbie/accessors.py#L310)
def pick_points(
    ds: xr.Dataset,
    points: pd.DataFrame,
    config,  # TODO fix the circular import for config.py or figure out where i can put this function
    method: Literal["nearest", "weighted"] = "nearest",
    k: Optional[int] = None,
    max_distance: Union[int, float] = 500,
    use_cached_tree: Union[bool, Literal["replant"]] = True,
    tree_name: Optional[str] = None,
) -> xr.Dataset:
    """Pick nearest neighbor grid values at selected points.

    Parameters
    ----------
    points : Pandas DataFrame
        A DataFrame with columns 'latitude' and 'longitude'
        representing the points to match to the model grid.
    method : {'nearest', 'weighted'}
        Method used to pick points.
        - `nearest` : Gets grid value nearest the requested point.
        - `weighted`: Gets four grid value nearest the requested
            point and compute the inverse-distance-weighted mean.
    k : None or int
        If None and method is nearest, `k=1`.
        If None and method is weighted, `k=4`.
        Else, specify the number of neighbors to find.
    max_distance : int or float
        Maximum distance in kilometers allowed for nearest neighbor
        search. Default is 500 km, which is very generous for any
        model grid. This can help the case when a requested point
        is off the grid.
    use_cached_tree : {True, False, "replant"}
        Controls if the BallTree object is caches for later use.
        By "plant", I mean, "create a new BallTree object."
        - `True` : Plant+save BallTree if it doesn't exist; load
            saved BallTree if one exists.
        - `False`: Plant the BallTree, even if one exists.
        - `"replant"` : Plant a new BallTree and save a new pickle.
    tree_name : str
        If None, use the ds.model and domain size as the tree's name.
        If ds.model does not exists, then the BallTree will not be
        cached, unless you provide the tree_name.

    Examples
    --------
    >>> H = Herbie("2024-03-28 00:00", model="hrrr")
    >>> ds = H.xarray("TMP:[5,6,7,8,9][0,5]0 mb", remove_grib=False)
    >>> points = pd.DataFrame(
    ...     {
    ...         "longitude": [-100, -105, -98.4],
    ...         "latitude": [40, 29, 42.3],
    ...         "stid": ["aa", "bb", "cc"],
    ...     }
    ... )

    Pick value at the nearest neighbor point
    >>> dsp = ds.pick_points(points, method="nearest")

    Get the weighted mean of the four nearest neighbor points
    >>> dsp = ds.pick_points(points, method="weighted")

    A Dataset is returned of the original grid reduced to the
    requested points, with the values from the `points` dataset
    added as new coordinates.

    A user can easily convert the result to a Pandas DataFrame
    >>> dsp.to_dataframe()

    If you want to select points by a station name, swap the
    dimension.
    >>> dsp = dsp.swap_dims({"point": "point_stid"})
    """
    try:
        from sklearn.neighbors import BallTree
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "scikit-learn is an 'extra' requirement, please use "
            "`pip install 'herbie-data[extras]'` for the full functionality."
        )

    def plant_tree(save_pickle: Optional[Union[Path, str]] = None):
        """Grow a new BallTree object from seedling."""
        timer = pd.Timestamp("now")
        print("INFO: 🌱 Growing new BallTree...", end="")
        tree = BallTree(np.deg2rad(df_grid), metric="haversine")
        print(
            f"🌳 BallTree grew in {(pd.Timestamp('now') - timer).total_seconds():.2} seconds."
        )
        if save_pickle:
            try:
                Path(save_pickle).parent.mkdir(parents=True, exist_ok=True)
                with open(save_pickle, "wb") as f:
                    pickle.dump(tree, f)
                print(f"INFO: Saved BallTree to {save_pickle}")
            except OSError:
                print(f"ERROR: Could not save BallTree to {save_pickle}.")
        return tree

    # ---------------------
    # Validate points input
    if ("latitude" not in points) and ("longitude" not in points):
        raise ValueError(
            "`points` DataFrame must have columns 'latitude' and 'longitude'"
        )

    if not all(points.latitude.between(-90, 90, inclusive="both")):
        raise ValueError("All latitude points must be [-90,90]")

    if not all(points.longitude.between(0, 360, inclusive="both")):
        if not all(points.longitude.between(-180, 180, inclusive="both")):
            raise ValueError("All longitude points must be [-180,180] or [0,360]")

    # ---------------------
    # Validate method input
    _method = set(["nearest", "weighted"])

    if method == "nearest" and k is None:
        # Get the value at the nearest grid point using BallTree
        k = 1
    elif method == "weighted" and k is None:
        # Compute the value of each variable from the inverse-
        # weighted distance of the values of the four nearest
        # neighbors.
        k = 4
    elif method in _method and isinstance(k, int):
        # Get the k nearest neighbors and return the values (nearest)
        # or compute the distance-weighted mean (weighted).
        pass
    else:
        raise ValueError(
            f"`method` must be one of {_method} and `k` must be an int or None."
        )

    # Only consider variables that have dimensions.
    ds = ds[[i for i in ds if ds[i].dims != ()]]

    if "latitude" in ds.dims and "longitude" in ds.dims:
        # Rename dims to x and y
        # This is needed for regular latitude-longitude grids like
        # GFS and IFS model data.
        ds = ds.rename_dims({"latitude": "y", "longitude": "x"})

    # Get Dataset's lat/lon grid and coordinate indices as a DataFrame.
    df_grid = (
        ds[["latitude", "longitude"]]
        .drop_vars([i for i, j in ds.coords.items() if not j.ndim])
        .to_dataframe()
    )

    # ---------------
    # BallTree object
    # Plant, plant+Save, or load

    if tree_name is None:
        tree_name = getattr(ds, "model", "UNKNOWN")

    if use_cached_tree and tree_name == "UNKNOWN":
        use_cached_tree = False
        print(
            "WARNING: Herbie won't cache the BallTree because it\n"
            "         doesn't know what to name it. Please specify\n"
            "         `tree_name` to cache the tree for use later."
        )

    pkl_BallTree_file = (
        Path(config.cache_dir).absolute()
        / "BallTree"
        / f"{tree_name}_{ds.x.size}-{ds.y.size}.pkl"
    )

    if not use_cached_tree:
        # Create a new BallTree. Do not save pickle.
        tree = plant_tree(save_pickle=None)
    elif use_cached_tree == "replant" or not pkl_BallTree_file.exists():
        # Create a new BallTree and save pickle.
        tree = plant_tree(save_pickle=pkl_BallTree_file)
    elif use_cached_tree:
        # Load BallTree from pickle.
        with open(pkl_BallTree_file, "rb") as f:
            tree = pickle.load(f)

    # -------------------------------------
    # Query points to find nearest neighbor
    # Note: Order matters, and lat/long must be in radians.
    # TODO: Maybe add option to use MultiProcessing here, to split
    # TODO:   the Dataset into chunks; or maybe not needed because
    # TODO:   the method is fast enough without the added complexity.
    dist, ind = tree.query(np.deg2rad(points[["latitude", "longitude"]]), k=k)

    # Convert distance to km by multiplying by the radius of the Earth
    dist *= 6371

    # Pick grid values for each value of k
    k_points = []
    df_grid = df_grid.reset_index()
    for i in range(k):
        a = points.copy()
        a["point_grid_distance"] = dist[:, i]
        a["grid_index"] = ind[:, i]

        a = pd.concat(
            [
                a,
                df_grid.iloc[a.grid_index].add_suffix("_grid").reset_index(drop=True),
            ],
            axis=1,
        )
        a.index.name = "point"

        if max_distance:
            flagged = a.loc[a.point_grid_distance > max_distance]
            a = a.loc[a.point_grid_distance <= max_distance]
            if len(flagged):
                print(
                    f"WARNING: {len(flagged)} points removed for exceeding {max_distance=} km threshold."
                )
                print(f"{flagged}")
                print("")

        # Get corresponding values from xarray
        # https://docs.xarray.dev/en/stable/user-guide/indexing.html#more-advanced-indexing
        ds_points = ds.sel(
            x=a.x_grid.to_xarray(),
            y=a.y_grid.to_xarray(),
        )
        ds_points.coords["point_grid_distance"] = a.point_grid_distance.to_xarray()
        ds_points["point_grid_distance"].attrs["long_name"] = (
            "Distance between requested point and nearest grid point."
        )
        ds_points["point_grid_distance"].attrs["units"] = "km"

        for i in points.columns:
            ds_points.coords[f"point_{i}"] = a[i].to_xarray()
            ds_points[f"point_{i}"].attrs["long_name"] = f"Requested grid point {i}"

        k_points.append(ds_points.drop_vars("point"))

    if method == "nearest" and k == 1:
        return k_points[0]

    elif method == "nearest" and k > 1:
        # New dimension k is the index of the n-th nearest neighbor
        return xr.concat(k_points, dim="k")

    elif method == "weighted":
        # Compute the inverse-distance weighted mean for each
        # variable from the four nearest points.
        b = xr.concat(k_points, dim="k")

        # Note: clipping accounts for the "divide by zero" case when
        # the requested point is exactly the nearest grid point.
        weights = (1 / b.point_grid_distance).clip(max=1e6)

        # Compute weighted mean of variables
        sum_of_weights = weights.sum(dim="k")
        weighted_sum = (b * weights).sum(dim="k")

        c = weighted_sum / sum_of_weights

        # Include some coordinates that were dropped as a result of
        # the line `weights.sum(dim='k')`.
        c.coords["latitude"] = b.coords["latitude"]
        c.coords["longitude"] = b.coords["longitude"]
        c.coords["point_grid_distance"] = b.coords["point_grid_distance"]

        return c
    else:
        raise ValueError("I didn't expect to be here.")
