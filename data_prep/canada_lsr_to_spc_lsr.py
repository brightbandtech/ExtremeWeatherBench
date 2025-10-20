# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

# %%
from extremeweatherbench import inputs

# %%
lsr = inputs.LSR(
    source="gs://extremeweatherbench/datasets/lsr_01012020_09272025.parq",
    variables=["report"],
    variable_mapping={"report": "reports"},
    storage_options={"anon": True},
)

# %%
lsr_df = lsr.open_and_maybe_preprocess_data_from_source()
lsr_df.rename(
    columns={
        "lat": "latitude",
        "lon": "longitude",
        "time": "valid_time",
        "Scale": "scale",
    },
    inplace=True,
)
lsr_df["scale"] = (lsr_df["scale"].replace("UNK", np.nan)).astype(float)


# %%
def convert_can_lsr_to_bb_lsr(can_lsr: pd.DataFrame) -> pd.DataFrame:
    """Convert the Canadian LSR data to the BB LSR data."""
    # rename to align with BB LSR column names
    modified_can_lsr = can_lsr.rename(
        columns={
            "Date/Time UTC": "valid_time",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )

    # time into datetime type
    modified_can_lsr["valid_time"] = pd.to_datetime(modified_can_lsr["valid_time"])

    # convert from cm to inch
    modified_can_lsr["hail_size"] = np.round(
        modified_can_lsr["Maximum Hail Dimension (mm)"] * 0.0393701 * 100, 0
    )
    # Convert EF scale strings to numeric values
    damage_mapping = {
        "ef0": 0,
        "default_ef0": 0,
        "ef1": 1,
        "ef2": 2,
        "ef3": 3,
        "ef4": 4,
        "ef5": 5,
    }
    modified_can_lsr["Damage"] = (
        modified_can_lsr["Damage"].map(damage_mapping).astype(float)
    )
    # merge hail size and Fujita scale into scale column, replacing NaNs with the other
    # column
    modified_can_lsr["scale"] = modified_can_lsr["hail_size"].fillna(
        modified_can_lsr["Damage"]
    )

    modified_can_lsr = modified_can_lsr[
        ["latitude", "longitude", "report_type", "valid_time", "scale"]
    ]
    return modified_can_lsr


# %%
can_lsr = pd.read_csv("gs://extremeweatherbench/deprecated/CanadaLSRData_2020-2024.csv")

converted_can_lsr = convert_can_lsr_to_bb_lsr(can_lsr)
combined_lsr_df = pd.concat([lsr_df, converted_can_lsr])
combined_lsr_df["latitude"] = combined_lsr_df["latitude"].astype(float)
combined_lsr_df["longitude"] = combined_lsr_df["longitude"].astype(float)

# %%
combined_lsr_df = combined_lsr_df.sort_values(by="valid_time")

# %%
combined_lsr_df

# %%
combined_lsr_df.to_parquet("combined_canada_australia_us_lsr_01012020_09272025.parq")
