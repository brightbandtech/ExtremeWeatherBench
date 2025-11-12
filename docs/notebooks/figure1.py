# figure 1: show all cases on a world map

# setup all the imports
import matplotlib.font_manager
import matplotlib.pyplot as plt

flist = matplotlib.font_manager.get_font_names()
from pathlib import Path  # noqa: E402

from extremeweatherbench import cases, defaults, evaluate  # noqa: E402

# make the basepath - change this to your local path
basepath = Path.home() / "ExtremeWeatherBench" / ""
basepath = str(basepath) + "/"

# ugly hack to load in our plotting scripts
import sys  # noqa: E402

sys.path.append(basepath + "/docs/notebooks/")
import paper_plotting as pp  # noqa: E402

# load in all of the events in the yaml file
print("loading in the events yaml file")
ewb_cases = cases.load_ewb_events_yaml_into_case_collection()
# build out all of the expected data to evalate the case
# this will not be a 1-1 mapping with ewb_cases because there are multiple data sources
# to evaluate for some cases
# for example, a heat/cold case will have both a case operator for ERA-5 data and GHCN
case_operators = cases.build_case_operators(
    ewb_cases, defaults.get_brightband_evaluation_objects()
)

# to plot the targets, we need to run the pipeline for each case and target
from joblib import Parallel, delayed  # noqa: E402
from joblib.externals.loky import get_reusable_executor  # noqa: E402

# load in all the case info (note this takes awhile in non-parallel form as it has to
# run all the target information for each case)
# this will return a list of tuples with the case id and the target dataset

print("running the pipeline for each case and target")
parallel = Parallel(n_jobs=8, return_as="generator", backend="loky")
case_operators_with_targets_established_generator = parallel(
    delayed(
        lambda co: (
            co.case_metadata.case_id_number,
            evaluate.run_pipeline(co.case_metadata, co.target),
        )
    )(case_operator)
    for case_operator in case_operators
)
case_operators_with_targets_established = list(
    case_operators_with_targets_established_generator
)
# this will throw a bunch of errors below but they're not consequential. this releases
# the memory as it shuts down the workers
# now that they're not used
get_reusable_executor().shutdown(wait=True)


n_rows = 5
n_cols = 3
col_space = 0.5
row_space = 0.5
figsize = (5 * n_cols + col_space * (n_cols - 1), 5 * n_rows + row_space * (n_rows - 1))
print(figsize)

fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

# the left hand column of figure one shows all of the cases for each event type
# plot the cases for each event type
print("plotting the cases for each event type")
pp.plot_all_cases(
    ewb_cases,
    event_type="tropical_cyclone",
    fill_boxes=True,
    ax=axs[0, 0],
)
pp.plot_all_cases(
    ewb_cases,
    event_type="freeze",
    fill_boxes=True,
    ax=axs[1, 0],
)
pp.plot_all_cases(
    ewb_cases,
    event_type="heat_wave",
    fill_boxes=True,
    ax=axs[2, 0],
)
pp.plot_all_cases(
    ewb_cases,
    event_type="atmospheric_river",
    fill_boxes=True,
    ax=axs[3, 0],
)
pp.plot_all_cases(
    ewb_cases,
    event_type="severe_convection",
    fill_boxes=True,
    ax=axs[4, 0],
)

# # the next column of figure one shows the cases for each event type with the obs
# # plot the cases for each event type with the observations
# pp.plot_all_cases_and_obs(
#     ewb_cases,
#     event_type="tropical_cyclone",
#     filename=basepath + "docs/notebooks/figs/ewb_tcs_obs.png",
#     targets=case_operators_with_targets_established,
# )
# pp.plot_all_cases_and_obs(
#     ewb_cases,
#     event_type="freeze",
#     filename=basepath + "docs/notebooks/figs/ewb_freeze_obs.png",
#     targets=case_operators_with_targets_established,
# )
# pp.plot_all_cases_and_obs(
#     ewb_cases,
#     event_type="heat_wave",
#     filename=basepath + "docs/notebooks/figs/ewb_heat_obs.png",
#     targets=case_operators_with_targets_established,
# )
# pp.plot_all_cases_and_obs(
#     ewb_cases,
#     event_type="atmospheric_river",
#     filename=basepath + "docs/notebooks/figs/ewb_ar_obs.png",
#     targets=case_operators_with_targets_established,
# )
# pp.plot_all_cases_and_obs(
#     ewb_cases,
#     event_type="severe_convection",
#     filename=basepath + "docs/notebooks/figs/ewb_convective_obs.png",
#     targets=case_operators_with_targets_established,
# )


fig.tight_layout()
fig.savefig(basepath + "docs/notebooks/figs/figure1.png")
