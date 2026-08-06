"""Input fields for the event detectors."""

import numpy as np
import pandas as pd
import xarray as xr


def make_ar_input_dataset(
    time_dim="valid_time",
    n_time=3,
    n_lat=40,
    n_lon=40,
    extra_dim=None,
    n_extra=1,
    blobs=(),
    chunk=True,
):
    """IVT and Laplacian fields with rectangular blobs above both thresholds.

    Each blob is (time_index, extra_index, lat_slice, lon_slice). Values are
    set so the AR criteria are met exactly inside the blob and nowhere else,
    which makes the resulting feature sizes predictable.
    """
    lat = np.linspace(20.0, 59.0, n_lat)
    lon = np.linspace(-160.0, -121.0, n_lon)

    dims = [time_dim, "latitude", "longitude"]
    shape = [n_time, n_lat, n_lon]
    coords = {
        time_dim: (
            pd.date_range("2023-01-01", periods=n_time, freq="6h")
            if time_dim == "valid_time"
            else pd.to_timedelta(np.arange(n_time) * 6, unit="h")
        ),
        "latitude": lat,
        "longitude": lon,
    }
    if extra_dim is not None:
        dims.insert(0, extra_dim)
        shape.insert(0, n_extra)
        coords[extra_dim] = pd.date_range("2023-01-01", periods=n_extra, freq="D")

    ivt_values = np.zeros(shape)
    lap_values = np.zeros(shape)
    for blob in blobs:
        t_idx, extra_idx, lat_slice, lon_slice = blob
        index: tuple = (t_idx, lat_slice, lon_slice)
        if extra_dim is not None:
            index = (extra_idx,) + index
        ivt_values[index] = 800.0
        lap_values[index] = 5.0

    ivt = xr.DataArray(ivt_values, dims=dims, coords=coords)
    lap = xr.DataArray(lap_values, dims=dims, coords=coords)
    if chunk:
        chunking = {extra_dim: 1} if extra_dim is not None else {time_dim: -1}
        ivt, lap = ivt.chunk(chunking), lap.chunk(chunking)
    return ivt, lap
