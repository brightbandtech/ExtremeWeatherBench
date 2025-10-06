"""Generate a icechunk store for all of the the CIRA data.

Data is available at:
https://noaa-oar-mlwp-data.s3.amazonaws.com/index.html

Data is generated as of 2025-10-04.
"""

import logging
import re
import warnings
from collections import defaultdict
from typing import Union

import fsspec
import icechunk
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
    file: str, fs: fsspec.filesystem, min_size: float = 7000000000
) -> Union[str, None]:
    """Find broken files in the CIRA data using a minimum file size.

    Args:
        file: The file to check
        fs: The fsspecfilesystem to check the file on
        min_size: The minimum file size to consider in bytes

    Returns: The file if it is not broken, otherwise None.
    """
    if "gfs" in file.split("/")[-1]:
        return None
    if fs.du(file) < min_size:
        return None
    return file


def remove_virtual_datasets_with_zlib_compression(vds: xr.Dataset):
    """Remove virtual datasets with zlib compression.

    Args:
        vds: The xarray dataset to check

    Returns: The dataset is not compressed with zlib, otherwise the dataset with the
    zlib compression removed.
    """
    if vds.variables["z"]._data.zarray.codec.filters is not None:
        return vds
    return None


def get_cira_data_urls() -> list[str]:
    """Get the URLs for all of the the CIRA data."""

    fs = fsspec.filesystem("s3", anon=True)
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


def generate_icechunk_store(vdt: xr.DataTree):
    """Generate a icechunk store for all of the the CIRA data."""
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "s3://noaa-oar-mlwp-data/",
            store=icechunk.s3_store(region="us-east-1", anonymous=True),
        ),
    )

    config.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(
            max_total_refs=100_000_000,
            preload_if=icechunk.ManifestPreloadCondition.name_matches(
                ".*time|.*latitude|.*longitude|.*level"
            ),
        ),
    )
    # create an in-memory icechunk repository that includes the virtual chunk containers
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage, config)

    # open a writable icechunk session to be able to add new contents to the store
    session = repo.writable_session("main")

    # write the virtual dataset to the session's IcechunkStore instance, using
    # VirtualiZarr's `.vz` accessor
    vdt.vz.to_icechunk(session.store)

    # commit your changes so that they are permanently available as a new snapshot
    snapshot_id = session.commit("Wrote first dataset")
    print(snapshot_id)

    # optionally persist the new permissions to be permanent, which you probably want
    # otherwise every user who wants to read the referenced virtual data back later
    # will have to repeat the `config.set_virtual_chunk_container` step at read time.
    repo.save_config()


def main():
    """Main function to generate a icechunk store for all of the the CIRA data."""
    # fs_local = fsspec.filesystem("")
    fs = fsspec.filesystem("s3", anon=True)
    data_urls = get_cira_data_urls(fs)

    parser = virtualizarr.parsers.HDFParser()
    registry = set_up_remote_store_and_registry("s3://noaa-oar-mlwp-data")
    parallel = joblib.Parallel(n_jobs=-1)
    # Group data URLs by model (the *FS part in the URI) using defaultdic
    grouped_urls = defaultdict(list)
    for url in data_urls:
        # Extract model from pattern s3://noaa-oar-mlwp-data/*FS/**/*.nc
        match = re.search(r"/([^/]*FS)/", url)
        if match:
            model = match.group(1)
            grouped_urls[model].append(url)

    # Process each model group and collect virtual datasets
    all_results = {}
    for model, model_urls in tqdm(grouped_urls.items()):
        print(f"Processing model: {model}")
        model_urls = [find_broken_files(model_url, fs) for model_url in model_urls[0:5]]
        model_urls = [n for n in model_urls if n is not None]
        results = parallel(
            joblib.delayed(open_virtual_dataset)(file, parser, registry)
            for file in model_urls
        )
        generator = (n for n in results)
        # all_results[model] = xr.concat(generator, dim="init_time", coords='minimal',
        # compat='override',combine_attrs='override')
        all_results[model] = generator
    return all_results


if __name__ == "__main__":
    results = main()
