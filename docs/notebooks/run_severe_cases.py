# setup all the imports
import seaborn as sns

from extremeweatherbench import (
    cases,
    defaults,
    derived,
    evaluate,
    inputs,
    metrics,
)

sns.set_theme(style="whitegrid")
from pathlib import Path  # noqa: E402

# make the basepath - change this to your local path
basepath = Path.home() / "ExtremeWeatherBench" / ""
basepath = str(basepath) + "/"

# ugly hack to load in our plotting scripts
import sys  # noqa: E402

sys.path.append(basepath + "/docs/notebooks/")

# setup the templates to load in the data

cira_severe_convection_forecast_FOURV2_GFS = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.CravenBrooksSignificantSevere()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    name="CIRA FOURv2 GFS",
    preprocess=defaults._preprocess_bb_severe_cira_forecast_dataset,
)

cira_severe_convection_forecast_GC_GFS = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/GRAP_v100_GFS.parq",
    variables=[derived.CravenBrooksSignificantSevere()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    name="CIRA GC GFS",
    preprocess=defaults._preprocess_bb_severe_cira_forecast_dataset,
)

cira_severe_convection_forecast_PANG_GFS = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/PANG_v100_GFS.parq",
    variables=[derived.CravenBrooksSignificantSevere()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    name="CIRA PANG GFS",
    preprocess=defaults._preprocess_bb_severe_cira_forecast_dataset,
)

hres_severe_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.CravenBrooksSignificantSevere()],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
    name="ECMWF HRES",
)

# Define threshold metrics
pph_metrics = [
    metrics.ThresholdMetric(
        metrics=[
            metrics.CriticalSuccessIndex(),
            metrics.FalseAlarmRatio(),
        ],
        forecast_threshold=15000,
        target_threshold=0.3,
    ),
    metrics.EarlySignal(threshold=15000),
]

# Define LSR metrics
lsr_metrics = [
    metrics.ThresholdMetric(
        metrics=[
            metrics.TruePositives(),
            metrics.FalseNegatives(),
        ],
        forecast_threshold=15000,
        target_threshold=0.5,
    )
]

FOURv2_SEVERE_EVALUATION_OBJECTS = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=lsr_metrics,
        target=defaults.lsr_target,
        forecast=cira_severe_convection_forecast_FOURV2_GFS,
    ),
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=pph_metrics,
        target=defaults.pph_target,
        forecast=cira_severe_convection_forecast_FOURV2_GFS,
    ),
]

GC_SEVERE_EVALUATION_OBJECTS = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=lsr_metrics,
        target=defaults.lsr_target,
        forecast=cira_severe_convection_forecast_GC_GFS,
    ),
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=pph_metrics,
        target=defaults.pph_target,
        forecast=cira_severe_convection_forecast_GC_GFS,
    ),
]

PANG_SEVERE_EVALUATION_OBJECTS = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=lsr_metrics,
        target=defaults.lsr_target,
        forecast=cira_severe_convection_forecast_PANG_GFS,
    ),
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=pph_metrics,
        target=defaults.pph_target,
        forecast=cira_severe_convection_forecast_PANG_GFS,
    ),
]

HRES_SEVERE_EVALUATION_OBJECTS = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=lsr_metrics,
        target=defaults.lsr_target,
        forecast=hres_severe_forecast,
    ),
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=pph_metrics,
        target=defaults.pph_target,
        forecast=hres_severe_forecast,
    ),
]

# load in all of the events in the yaml file
ewb_cases = cases.load_ewb_events_yaml_into_case_collection()
ewb_cases = ewb_cases.select_cases("event_type", "severe_convection")

ewb_fourv2 = evaluate.ExtremeWeatherBench(ewb_cases, FOURv2_SEVERE_EVALUATION_OBJECTS)
ewb_gc = evaluate.ExtremeWeatherBench(ewb_cases, GC_SEVERE_EVALUATION_OBJECTS)
ewb_pang = evaluate.ExtremeWeatherBench(ewb_cases, PANG_SEVERE_EVALUATION_OBJECTS)
ewb_hres = evaluate.ExtremeWeatherBench(ewb_cases, HRES_SEVERE_EVALUATION_OBJECTS)

parallel_config = {"backend": "loky", "n_jobs": 48}

<<<<<<< Updated upstream
# fourv2_results = ewb_fourv2.run(parallel_config=parallel_config)
=======
fourv2_results = ewb_fourv2.run(parallel_config=parallel_config)
>>>>>>> Stashed changes
# gc_results = ewb_gc.run(parallel_config=parallel_config)
# pang_results = ewb_pang.run(parallel_config=parallel_config)
# hres_results = ewb_hres.run(parallel_config=parallel_config)

<<<<<<< Updated upstream
# fourv2_results.to_pickle(basepath + "docs/notebooks/figs/fourv2_severe_results.pkl")
=======
fourv2_results.to_pickle(basepath + "docs/notebooks/figs/fourv2_severe_results.pkl")
>>>>>>> Stashed changes
# gc_results.to_pickle(basepath + "docs/notebooks/figs/gc_severe_results.pkl")
# pang_results.to_pickle(basepath + "docs/notebooks/figs/pang_severe_results.pkl")
# hres_results.to_pickle(basepath + "docs/notebooks/figs/hres_severe_results.pkl")
