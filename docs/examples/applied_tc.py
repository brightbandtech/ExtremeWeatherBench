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
import logging

# %%
# %%
from extremeweatherbench import cases, derived, evaluate, inputs, metrics

# %%
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
case_yaml = cases.load_ewb_events_yaml_into_case_collection()
case_yaml.select_cases(by="case_id_number", value=200)

# %%
ibtracs_target = inputs.IBTrACS()

# %%
hres_forecast = inputs.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.TropicalCycloneTrackVariables],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

# %%
# just one for now
tc_evaluation_object = [
    inputs.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=[
            metrics.LandfallTimeME,
            metrics.LandfallIntensityMAE,
            metrics.LandfallDisplacement,
        ],
        target=ibtracs_target,
        forecast=hres_forecast,
    ),
]
# %%
test_ewb = evaluate.ExtremeWeatherBench(
    case_metadata=case_yaml,
    evaluation_objects=tc_evaluation_object,
)
logger.info("Starting EWB run")
outputs = test_ewb.run(
    n_jobs=24,
)
outputs.to_csv("tc_metric_test_results.csv")
