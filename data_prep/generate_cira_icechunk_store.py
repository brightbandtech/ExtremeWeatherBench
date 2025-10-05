"""Generate a icechunk store for all of the the CIRA data.

Data is available at:
https://noaa-oar-mlwp-data.s3.amazonaws.com/index.html

Data is generated as of 2025-10-04.
"""

import logging
import warnings
from typing import Union

import fsspec
import joblib
import virtualizarr
import xarray as xr
from obstore.store import from_url
from tqdm.auto import tqdm
from virtualizarr.registry import ObjectStoreRegistry

warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification*",
    category=UserWarning,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_broken_files(
    file: str, fs: fsspec.filesystem, min_size: float = 1000000000
) -> Union[str, None]:
    """Find broken files in the CIRA data using a minimum file size.

    Args:
        file: The file to check
        fs: The fsspecfilesystem to check the file on
        min_size: The minimum file size to consider in bytes

    Returns: The file if it is broken, otherwise None.
    """
    if fs.du(file) < min_size:
        return file
    return None


def get_cira_data_urls() -> list[str]:
    """Get the URLs for all of the the CIRA data."""

    fs = fsspec.filesystem("s3", anon=True)
    # models = [
    #     "AURO_v100_GFS",
    #     "AURO_v100_IFS",
    #     "FOUR_v100_GFS",
    #     "FOUR_v200_GFS",
    #     "FOUR_v200_IFS",
    #     "GRAP_v100_GFS",
    #     "GRAP_v100_IFS",
    #     "PANG_v100_GFS",
    #     "PANG_v100_IFS",
    # ]
    flist = fs.glob("s3://noaa-oar-mlwp-data/*FS/**/*.nc")
    flist = sorted(["s3://" + f for f in flist])
    return flist


def set_up_remote_store_and_registry(
    bucket: str, region: str = "us-east-1"
) -> ObjectStoreRegistry:
    """Set up a remote store and registry for the CIRA data.

    Args:
        bucket: The bucket to use for the remote store with approriate URI prefix
        e.g. "s3://noaa-oar-mlwp-data"
        region: The region the remote store is located in

    Returns: The remote registry
    """
    store = from_url(bucket, region=region, skip_signature=True)
    registry = ObjectStoreRegistry({bucket: store})
    return registry


def open_virtual_dataset(
    file: str,
    parser: virtualizarr.parsers.HDFParser,
    registry: virtualizarr.registry.ObjectStoreRegistry,
) -> xr.Dataset:
    """Open a virtual dataset from a file using virtualizarr.

    Args:
        file: The file to open
        parser: The parser to use to parse the file
        registry: The registry to use to access the file

    Returns: The virtual dataset as an xarray dataset
    """
    return virtualizarr.open_virtual_dataset(file, parser=parser, registry=registry)


def generate_icechunk_store(data_urls: list[str]):
    """Generate a icechunk store for all of the the CIRA data."""
    pass


def main():
    """Main function to generate a icechunk store for all of the the CIRA data."""
    # fs_local = fsspec.filesystem("")
    data_urls = get_cira_data_urls()

    parser = virtualizarr.parsers.HDFParser()
    registry = set_up_remote_store_and_registry("s3://noaa-oar-mlwp-data")
    parallel = joblib.Parallel(n_jobs=-1)
    results = parallel(
        joblib.delayed(open_virtual_dataset)(file, parser, registry)
        for file in tqdm(data_urls[0:10])
    )

    return results


if __name__ == "__main__":
    results = main()
    results[0]
