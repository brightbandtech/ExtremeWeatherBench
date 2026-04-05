# Data Preparation

The scripts in `data_prep/` regenerate EWB's source datasets: case bounding
boxes, observation archives, and model stores. Most users never need to run
them. Run them when you need to extend the case set, update an observation
archive, or rebuild a data store from scratch.

All scripts must be run from the repository root.

---

## Plot Temperature Events

**File:** `data_prep/plot_temperature_events.py`

Plots the maximum number of consecutive heat wave or cold snap days for a
single case from `events.yaml`. Auto-detects event type from the case record.
Also exports `max_consecutive_days` and `plot_consecutive_map`, which
`heat_cold_bounds_global.py` and `heat_cold_bounds_case.py` import directly.

**Usage**

```bash
python data_prep/plot_temperature_events.py \
    --case-id-number 2 \
    --output case_2_consecutive_heatwave_days.png
```

**Output**

One PNG at the path given by `--output`. Default filename when `--output` is
omitted: `case_N_consecutive_{heatwave|cold_snap}_days.png` in the current
directory.

---

## Heat / Cold Bounds — Global Detection

**File:** `data_prep/heat_cold_bounds_global.py`

Scans ERA5 2 m temperature over a date range and detects heat wave and cold
snap events globally over land. A heat wave requires daily max > 85th
percentile for 3+ consecutive days; a cold snap requires daily min < 15th
percentile. Spatiotemporal blobs are tracked and terminated when their area
drops below 50 % of peak. Produces a CSV of bounding boxes and two PNG maps.

**Usage**

```bash
python data_prep/heat_cold_bounds_global.py \
    --start-date 2023-06-01 \
    --end-date 2023-09-01 \
    --output heat_cold_global.csv \
    --n-workers 4
```

**Output**

CSV at `--output` with columns `label`, `event_type`, `start_date`,
`end_date`, `latitude_min/max`, `longitude_min/max`. Two PNG maps saved
alongside it: `<stem>_heatwave.png` and `<stem>_cold_snap.png`.

---

## Heat / Cold Bounds — Case Validation

**File:** `data_prep/heat_cold_bounds_case.py`

For each heat wave or cold snap case in `events.yaml`, iteratively expands the
existing bounding box by 2° per side until fewer than 50 % of edge grid points
exceed the climatological threshold, or 10 iterations are reached. Processes
all cases in parallel via joblib.

**Usage**

```bash
python data_prep/heat_cold_bounds_case.py \
    --output heat_cold_yaml.csv \
    --n-workers 4
```

**Output**

CSV at `--output` with final bounding boxes. One PNG per case saved to the
same directory as `--output`, named
`case_<id>_consecutive_{heatwave|cold_snap}_days.png`.

---

## Generate GHCNh

**File:** `data_prep/generate_ghcnh.py`

Downloads GHCNh station data for 2020–2024 from NCEI, aggregates to hourly
resolution, applies QC filtering, and appends to a single parquet file.
Already-processed station-year combinations are skipped on re-run. Up to 1000
concurrent downloads via `asyncio`.

**Dependencies**

```
pip install aiohttp nest_asyncio
```

**Usage**

```bash
python data_prep/generate_ghcnh.py
```

**Output**

`ghcnh_all_2020_2024.parq` in the current directory.

---

## AR Bounds

**File:** `data_prep/ar_bounds.py`

Calculates bounding boxes for atmospheric river cases from `events.yaml`.
Runs IVT-based AR detection on ERA5, identifies the largest AR object per
case using connected-component labelling, and adds a spatial buffer. Processes
cases in parallel (8 workers by default). Requires a running Dask cluster
(local cluster started automatically).

**Usage**

```bash
python data_prep/ar_bounds.py
```

**Output**

`ar_bounds_results_enhanced.pkl` in the current directory. Load with
`pickle.load` — each element is a dict with keys `case_id`, `title`,
`ar_largest_object_bounds`, `buffered_bounds`, and diagnostics.

---

## IBTrACS Bounds

**File:** `data_prep/ibtracs_bounds.py`

Downloads IBTrACS CSV from NCEI, computes a track-based bounding box for each
tropical cyclone case in `events.yaml`, and writes the updated bounds back to
the installed package's `events.yaml` in place. Logs which cases were changed.

**Usage**

```bash
python data_prep/ibtracs_bounds.py
```

**Output**

Modifies `src/extremeweatherbench/data/events.yaml` in place. No separate
output file is created.

---

## Severe Convection Bounds

**File:** `data_prep/severe_convection_bounds.py`

Creates bounding boxes around PPH non-zero regions for severe convection cases
from `events.yaml`. Applies a 250 km buffer by default. Requires a precomputed
PPH `DataArray` as input (produced by
`practically_perfect_hindcast_from_lsr.py`).

**Usage**

```python
from data_prep.severe_convection_bounds import main

bounding_boxes, df = main(
    pph_data="practically_perfect_hindcast_20200104_20250927.zarr",
    events_yaml_path="src/extremeweatherbench/data/events.yaml",
    output_path="data_prep/pph_severe_convection_bounding_boxes",
    buffer_km=250,
)
```

Or pass a PPH path directly from the command line:

```bash
python data_prep/severe_convection_bounds.py \
    practically_perfect_hindcast_20200104_20250927.zarr
```

**Output**

`<output_path>.csv` and `<output_path>.yaml` with bounding box records.

---

## Practically Perfect Hindcast from LSR

**File:** `data_prep/practically_perfect_hindcast_from_lsr.py`

Computes the Practically Perfect Hindcast (PPH) from Local Storm Report (LSR)
data using a Gaussian smoothing method (Hitchens et al. 2013). Reads from the
EWB public LSR target at `gs://extremeweatherbench`. Runs all valid times in
parallel via joblib. Stores results as a dense zarr archive.

**Usage**

```bash
python data_prep/practically_perfect_hindcast_from_lsr.py
```

**Output**

`practically_perfect_hindcast_20200104_20250927.zarr` in the current directory.

---

## Combined LSR Processing

**File:** `data_prep/combined_lsr_processing.py`

Downloads US Local Storm Report data from SPC NOAA for 2020–2025, adds
Canadian and Australian storm reports, and writes the combined dataset to
parquet. Runs verification checks on the output row count and date coverage.

**Dependencies**

```
pip install aiohttp
```

**Usage**

```bash
python data_prep/combined_lsr_processing.py
```

**Output**

`combined_canada_australia_us_lsr_01012020_09272025.parq` in the current
directory.

---

## CIRA Icechunk Generation

**File:** `data_prep/cira_icechunk_generation.py`

Builds an icechunk store for CIRA MLWP model data. Reads model files from
`s3://noaa-oar-mlwp-data` via VirtualiZarr and writes the resulting
`DataTree` to the EWB GCS bucket. Requires GCS write credentials for
`gs://extremeweatherbench`.

**Usage**

```bash
# Set application credentials in the script before running:
# storage = icechunk.gcs_storage(..., application_credentials="/path/to/creds.json")
python data_prep/cira_icechunk_generation.py
```

**Output**

Writes to the `cira-icechunk` prefix in the `extremeweatherbench` GCS bucket.

---

## Convert to Kerchunk

**File:** `data_prep/convert_to_kerchunk.py`

Provides two functions for converting CIRA MLWP NetCDF files from S3 to
kerchunk virtual references. `generate_json_from_nc` scans a single file and
writes a JSON; `xarray_dataset_from_json_list` combines a list of JSONs into
a single virtual zarr `Dataset`. No command-line entry point.

**Usage**

```python
import fsspec
from data_prep.convert_to_kerchunk import generate_json_from_nc, xarray_dataset_from_json_list

fs_read = fsspec.filesystem("s3", anon=True)
fs_out  = fsspec.filesystem("file")
so      = {"anon": True}

json_list = generate_json_from_nc(
    file_url="s3://noaa-oar-mlwp-data/FourCastNetv2/...",
    fs_read=fs_read,
    fs_out=fs_out,
    so=so,
    json_dir="/tmp/cira_jsons/",
)

ds = xarray_dataset_from_json_list(
    json_list=json_list,
    combined_json_directory="/tmp/cira_jsons/",
    fs_out=fs_out,
)
```

**Output**

Per-file JSON references in `json_dir` and a `combined.json` in
`combined_json_directory`. `xarray_dataset_from_json_list` returns an
`xr.Dataset` backed by the combined reference.

---

## Generate CAPE Reference Data

**File:** `data_prep/generate_cape_reference_data.py`

Fetches ERA5 atmospheric profiles from ARCO-ERA5, computes CAPE and CIN with
MetPy, and saves representative profiles as `.npz` files for unit testing.
Also generates synthetic pathological profiles covering edge cases. Requires
GCP application-default credentials and `uv`.

**Dependencies**

```
pip install metpy
```

Dependencies are also declared inline for `uv`:

```bash
gcloud auth application-default login
uv run data_prep/generate_cape_reference_data.py
```

**Output**

`tests/data/era5_reference.npz` and `tests/data/pathological_profiles.npz`.
