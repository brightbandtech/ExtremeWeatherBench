import warnings

import icechunk
import joblib
import numpy as np
import obstore as obs
import pandas as pd
import xarray as xr
from obstore.store import from_url
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

from extremeweatherbench import utils


def process(path, parser, registry):
    vds = open_virtual_dataset(
        url=path,
        parser=parser,
        registry=registry,
        decode_times=True,
        loadable_variables=["time", "latitude", "longitude", "level"],
    )
    return vds


def process_model(model_key, model_data):
    try:
        combined_vds = xr.combine_nested(
            model_data,
            concat_dim="init_time",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
            join="override",
        )
        combined_vds = combined_vds.rename({"time": "valid_time"})

        combined_vds = combined_vds.assign_coords(
            init_time=[
                pd.to_datetime(f.attrs["initialization_time"]) for f in model_data
            ],
        )

        # hard code lead times for now
        lead_times = np.linspace(0, 240, 41).astype("timedelta64[h]")
        combined_vds = combined_vds.assign_coords(lead_time=("valid_time", lead_times))
        combined_vds = combined_vds.swap_dims({"valid_time": "lead_time"})
        return model_key, combined_vds
    except Exception as e:
        print(e)
        print(model_key)
        return model_key, None


warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification*",
    category=UserWarning,
)


def main():
    bucket = "s3://noaa-oar-mlwp-data"
    store = from_url(bucket, region="us-east-1", skip_signature=True)
    prefix_list = [
        n for n in obs.list_with_delimiter(store)["common_prefixes"] if n.endswith("FS")
    ]

    registry = ObjectStoreRegistry({bucket: store})
    parser = HDFParser()

    model_dict = {}
    for model in prefix_list:
        stream = obs.list(store, model + "/", chunk_size=1)
        t = [n for n in stream]
        t = [n for ns in t for n in ns]
        stream = obs.list(store, model + "/", chunk_size=1)

        with joblib.parallel_config(**{"backend": "loky", "n_jobs": 300}):
            model_dict[model] = utils.ParallelTqdm(total=len(t))(
                # None is the cache_dir, we can't cache in parallel mode
                joblib.delayed(process)(
                    "s3://noaa-oar-mlwp-data/" + i[0]["path"], parser, registry
                )
                for i in stream
            )

    # runs after this date have a different chunking scheme, which cannot be combined for now (ZEP003)
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
        results = joblib.Parallel()(
            joblib.delayed(process_model)(model_key, single_chunk_model_dict[model_key])
            for model_key in single_chunk_model_dict.keys()
        )

    concat_model_dict = {
        model_key: result for model_key, result in results if result is not None
    }

    cira_datatree = xr.DataTree.from_dict(concat_model_dict)

    storage = icechunk.gcs_storage(
        bucket="extremeweatherbench", prefix="cira-icechunk", application_credentials=""
    )

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            url_prefix="s3://noaa-oar-mlwp-data/",
            store=icechunk.s3_store(region="us-east-1", anonymous=True),
        ),
    )

    repo = icechunk.Repository.create(storage, config)

    session = repo.writable_session(branch="main")

    cira_datatree.vz.to_icechunk(session.store)
    session.commit("drop in cira icechunk store")


if __name__ == "__main__":
    main()


# %%

test_storage = icechunk.gcs_storage(
    bucket="extremeweatherbench", prefix="cira-icechunk", anonymous=True
)
test_repo = icechunk.Repository.open(test_storage)
session = test_repo.readonly_session(branch="main")
dt = xr.open_datatree(session.store, engine="zarr")

# %%


# %%
