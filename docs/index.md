# ExtremeWeatherBench

## Introduction

Welcome to the official documentation for ExtremeWeatherBench, a benchmark dataset and evaluation framework for extreme weather event prediction. This guide will help you understand how to use the library effectively, explore its features, and integrate it into your projects.

Traditional meteorological evaluation often relies on grid-point-based metrics that average performance across space and time. However, for extreme weather events, these approaches can mask critical performance issues. ExtremeWeatherBench emphasizes event and case-based metrics that evaluate predictions at the level of discrete weather events. This approach better aligns with how forecasts are actually used by decision-makers and stakeholders. By focusing on user-driven metrics that directly measure a model's ability to predict the timing, location, and intensity of specific extreme events, we provide evaluations that are more relevant to real-world applications such as emergency management, resource allocation, and public safety warnings. This user-centric approach ensures that model improvements translate to meaningful societal benefits alongside statistical improvements.

## About ExtremeWeatherBench

ExtremeWeatherBench provides a standardized framework for evaluating machine learning models on extreme weather prediction tasks. It includes curated datasets, evaluation metrics, and baseline models specifically designed for extreme weather phenomena such as:

- Heat waves
- Freezes (cold snaps)
- Severe convection (hail and tornadoes)
- Tropical cyclones
- Atmospheric rivers

and is able to be extended with easily added components.


## Features

- **AIWP and Evaluation datasets**: 
    - Pre-processed AIWP forecasts ready for analysis [Radford et al. 2025](https://journals.ametsoc.org/view/journals/bams/106/1/BAMS-D-24-0057.1.xml) 
    - Curated global point observations from NCEI's [Global Historical Climatology Network (GHCN)](https://www.ncei.noaa.gov/news/next-generation-climate-dataset-built-seamless-integration)
    - Performant IBTrACS access via [Polars](https://pola.rs/) for tropical cyclones
    - Local storm reports for multiple countries, standardized into one source
    - Practically perfect hindcasts generated from local storm reports via zarr
    - Distinctive climatologies including 85th and 15th percentile surface temperatures via zarr

- **Standardized evaluation**: Consistent foundational metrics for fair model comparison
- **Easy to use**: Simple and intuitive API for researchers and practitioners, including a friendly learning curve to build bespoke metrics and integration of other forecast and target datasets
- **Well documented**: Comprehensive documentation with examples