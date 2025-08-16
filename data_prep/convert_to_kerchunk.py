"""Tooling to kerchunk CIRA data from
https://noaa-oar-mlwp-data.s3.amazonaws.com/index.html and convert it into
jsons for xarray datasets."""

import fsspec
import ujson
import xarray as xr
from kerchunk.combine import MultiZarrToZarr
from kerchunk.hdf import SingleHdf5ToZarr


def generate_json_from_nc(
    file_url: str,
    fs_read: fsspec.filesystem,
    fs_out: fsspec.filesystem,
    so: dict,
    json_dir: str,
) -> list:
    """Generate a kerchunk JSON file from a NetCDF file.

    Args:
        file_url: The URL/URI of the file to convert
        fs_read: The filesystem to read the file from
        fs_out: The filesystem to write the json to
        so: The storage options for the file
        json_dir: The directory to write the json to

    Returns a globbed list of jsons in the defined directory.
    """
    with fs_read.open(file_url, **so) as infile:
        h5chunks = SingleHdf5ToZarr(infile, file_url, inline_threshold=300)
        file_split = file_url.split(
            "/"
        )  # seperate file path to create a unique name for each json
        model = file_split[1].split("_")[0]
        date_string = file_split[-1].split("_")[3]
        outf = f"{json_dir}{model}_{date_string}.json"
        with fs_out.open(outf, "wb") as f:
            f.write(ujson.dumps(h5chunks.translate()).encode())

    json_list = fs_out.glob(f"{json_dir}*.json")
    return json_list


def xarray_dataset_from_json_list(
    json_list: list,
    combined_json_directory: str,
    fs_out: fsspec.filesystem,
    combined_json_name: str = "combined.json",
) -> xr.Dataset:
    """Combine the list of jsons into a single file.
    This is hardcoded to assume the CIRA conventions in the s3://noaa-oar-mlwp-data
    bucket, available at https://noaa-oar-mlwp-data.s3.amazonaws.com/index.html

    Arguments:
        json_list: A list of jsons in the directory to combine. Assumes full path
        combined_json_directory: The directory to write the combined json file to
        fs_out: the local (or remote) fsspec filesystem to write the combined json file
        with

    Returns an xarray dataset."""
    mzz = MultiZarrToZarr(
        json_list,
        remote_protocol="s3",
        remote_options={"anon": True},
        coo_map={"init_time": "attr:initialization_time"},
        concat_dims=["init_time"],
        identical_dims=["latitude", "longitude", "level"],
    )

    d = mzz.translate()

    with fs_out.open(f"{combined_json_directory}{combined_json_name}", "wb") as f:
        f.write(ujson.dumps(d).encode())

    backend_args = {
        "consolidated": False,
        "storage_options": {
            "fo": d,
            "remote_protocol": "s3",
            "remote_options": {"anon": True},
        },
    }
    return xr.open_dataset("reference://", engine="zarr", backend_kwargs=backend_args)
