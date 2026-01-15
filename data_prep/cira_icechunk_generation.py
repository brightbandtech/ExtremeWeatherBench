"""
This script is used to generate an icechunk store for the CIRA MLWP data.

Credit to CIRA for producing the AIWP model data, Tom Nicholas for VirtualiZarr,
the Earthmover team for icechunk.

To access the icechunk store, you can use the following code:

    import icechunk
    import xarray as xr

    test_storage = icechunk.gcs_storage(
        bucket="extremeweatherbench", prefix="cira-icechunk", anonymous=True
    )
    test_repo = icechunk.Repository.open(test_storage)
    session = test_repo.readonly_session(branch="main")
    dt = xr.open_datatree(session.store, engine="zarr")

Which will return a DataTree object with the CIRA MLWP data for the models (except
FCNv1, which is not compatible with this approach).
"""

import logging
import warnings

import icechunk
import joblib
import numpy as np
import obstore as obs
import pandas as pd
import virtualizarr
import virtualizarr.parsers
import virtualizarr.registry
import xarray as xr

from extremeweatherbench import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to suppress warnings about numcodecs codecs not being in the Zarr version 3
# specification. Warnings will still output when running it in parallel in joblib as
# there doesn't seem to be a way to suppress them in each newly spawned process.
warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification*",
    category=UserWarning,
)


def process_single_virtual_dataset(
    path: str,
    parser: virtualizarr.parsers.HDFParser,
    registry: virtualizarr.registry.ObjectStoreRegistry,
    loadable_variables: list[str] = ["time", "latitude", "longitude", "level"],
    decode_times: bool = True,
) -> xr.Dataset:
    """Process a single HDF/netCDF virtual dataset from a path.

    Args:
        path: The path to the virtual dataset (e.g. "s3://this/path/to/a/netcdf.nc").
        parser: The parser to use to parse the virtual dataset.
        registry: The virtualizarr.registry.ObjectStoreRegistry to use to access the virtual dataset.

    Returns:
        A Virtualizarr dataset
    """
    vds = virtualizarr.open_virtual_dataset(
        url=path,
        parser=parser,
        registry=registry,
        loadable_variables=loadable_variables,
        decode_times=decode_times,
    )
    return vds


def process_cira_model(
    model_key: str, model_data: list[xr.Dataset]
) -> tuple[str, xr.Dataset | None]:
    """Merge a list of singular virtual datasets into a single concatenated dataset.

    Args:
        model_key: The key of the model.
        model_data: A list of singular virtual datasets.

    Returns:
        A tuple of the model key and the concatenated dataset if successful, otherwise
            None.
    """

    # Some models e.g. FCNv1 are not compatible with this approach
    # Also, in the scenario of a problem like variable chunking (ZEP003), concatenation
    # will fail, so return None in that case.
    try:
        # Combine the virtual datasets into a single dataset. Args here are established
        # defaults for virtualizarr that also work for this case. When creating a new
        # dimension with concat_dim, join="override" and combine_attrs="drop_conflicts"
        # prevents fancy indexing errors.
        combined_vds = xr.combine_nested(
            model_data,
            concat_dim="init_time",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
            join="override",
        )

        # Rename the time coordinate to valid_time to be consistent with EWB conventions
        combined_vds = combined_vds.rename({"time": "valid_time"})

        # Assign the init_time attribute to the concatenated dataset
        combined_vds = combined_vds.assign_coords(
            init_time=[
                pd.to_datetime(f.attrs["initialization_time"]) for f in model_data
            ],
        )

        # Hard code lead times for now, inconsistent values in netcdf attributes
        lead_times = np.linspace(0, 240, 41).astype("timedelta64[h]")

        # Assign the lead time coordinate to the concatenated dataset
        combined_vds = combined_vds.assign_coords(lead_time=("valid_time", lead_times))

        # Swap the valid_time and lead_time dimensions to be consistent with EWB
        # conventions
        combined_vds = combined_vds.swap_dims({"valid_time": "lead_time"})

        # Return the model key and the concatenated dataset
        return model_key, combined_vds

    # If there is an error, log it and return a dict with value being None
    except Exception as e:
        logger.error(f"Error processing model {model_key}: {e}, returning None")
        return model_key, None


def generate_cira_icechunk_store():
    """Generate a CIRA icechunk store from the CIRA MLWP data."""

    # CIRA bucket URI
    bucket = "s3://noaa-oar-mlwp-data"

    # Build the ObjectStore from the URI, knowing the region and skipping signature
    store = obs.store.from_url(bucket, region="us-east-1", skip_signature=True)

    # Subset the prefixes to only include the model directories
    prefix_list = [
        n for n in obs.list_with_delimiter(store)["common_prefixes"] if n.endswith("FS")
    ]

    # Build the ObjectStoreRegistry and HDFParser
    registry = virtualizarr.registry.ObjectStoreRegistry({bucket: store})
    parser = virtualizarr.parsers.HDFParser()

    model_dict = {}
    for model in prefix_list:
        stream = obs.list(store, model + "/", chunk_size=1)
        t = [n for n in stream]
        t = [n for ns in t for n in ns]
        stream = obs.list(store, model + "/", chunk_size=1)

        with joblib.parallel_config(**{"backend": "loky", "n_jobs": -1}):
            model_dict[model] = utils.ParallelTqdm(total=len(t))(
                # None is the cache_dir, we can't cache in parallel mode
                joblib.delayed(process_single_virtual_dataset)(
                    "s3://noaa-oar-mlwp-data/" + i[0]["path"], parser, registry
                )
                for i in stream
            )

    # Runs starting 27 May 2025 have a different chunking scheme, which cannot be
    # concatenated for now (ZEP003)
    single_chunk_model_dict = {
        n: [
            item
            for item in model_dict[n]
            if item["time"][0] < pd.to_datetime("2025-05-27T00:00:00.000000000")
        ]
        for n in model_dict.keys()
    }

    concat_model_dict = {}

    with joblib.parallel_config(**{"backend": "loky", "n_jobs": -1}):
        results = utils.ParallelTqdm(total=len(single_chunk_model_dict))(
            joblib.delayed(process_cira_model)(
                model_key, single_chunk_model_dict[model_key]
            )
            for model_key in single_chunk_model_dict.keys()
        )

    # Filter out any None results and create a dictionary of model keys and concatenated
    # datasets
    concat_model_dict = {
        model_key: result for model_key, result in results if result is not None
    }

    # Create a DataTree from the dictionary of model keys and concatenated datasets
    cira_datatree = xr.DataTree.from_dict(concat_model_dict)

    # Build the GCS storage for the icechunk repository. This will fail if you do not
    # have the application credentials for write access to the EWB bucket. For another
    # GCS store, run gcloud auth application-default login and find where the generated
    # json credentials are stored, and use that path in application_credentials.
    storage = icechunk.gcs_storage(
        bucket="extremeweatherbench", prefix="cira-icechunk", application_credentials=""
    )

    # Build the RepositoryConfig with default config settings
    config = icechunk.RepositoryConfig.default()

    # Set the virtual chunk container to the CIRA bucket in S3
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            url_prefix="s3://noaa-oar-mlwp-data/",
            store=icechunk.s3_store(region="us-east-1", anonymous=True),
        ),
    )

    # Create the repository
    repo = icechunk.Repository.create(storage, config)

    # Create a writable session to the repository
    session = repo.writable_session(branch="main")

    # Convert the DataTree to icechunk and commit to the repository
    cira_datatree.vz.to_icechunk(session.store)

    # Commit the changes to the repository; required for icechunk to be used from the
    # GCS store
    session.commit("drop in cira icechunk store")


if __name__ == "__main__":
    generate_cira_icechunk_store()
