# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
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
    source=inputs.LSR_URI,
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
def convert_aus_lsr_to_bb_lsr(aus_lsr: pd.DataFrame) -> pd.DataFrame:
    """Convert the Australian LSR data to the BB LSR data."""
    # rename to align with BB LSR column names
    modified_aus_lsr = aus_lsr.rename(
        columns={
            "Date/Time UTC": "valid_time",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )

    # time into datetime type
    modified_aus_lsr["valid_time"] = pd.to_datetime(modified_aus_lsr["valid_time"])

    # convert from cm to inch
    modified_aus_lsr["hail_size"] = np.round(
        modified_aus_lsr["Hail size"] * 0.393701 * 100, 0
    )

    # merge hail size and Fujita scale into scale column, replacing NaNs with the other
    # column
    modified_aus_lsr["scale"] = modified_aus_lsr["hail_size"].fillna(
        modified_aus_lsr["Fujita scale"]
    )

    # drop unnecessary columns and reorder
    modified_aus_lsr = modified_aus_lsr.drop(
        columns=["hail_size", "Hail size", "Fujita scale", "Nearest town", "State"]
    )
    modified_aus_lsr = modified_aus_lsr[
        ["latitude", "longitude", "report_type", "valid_time", "scale"]
    ]
    return modified_aus_lsr


# %%
aus_lsr = pd.read_csv(
    "gs://extremeweatherbench/deprecated/AustralianLSRData_2020-2024.csv"
)

converted_aus_lsr = convert_aus_lsr_to_bb_lsr(aus_lsr)
lsr_df = pd.concat([lsr_df, converted_aus_lsr])
lsr_df["latitude"] = lsr_df["latitude"].astype(float)
lsr_df["longitude"] = lsr_df["longitude"].astype(float)
