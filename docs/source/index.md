# ExtremeWeatherBench

## Introduction

Welcome to the official documentation for ExtremeWeatherBench, a benchmark dataset and evaluation framework for extreme weather event prediction. This guide will help you understand how to use the library effectively, explore its features, and integrate it into your projects.

Traditional meteorological evaluation often relies on grid-point-based metrics that average performance across space and time. However, for extreme weather events, these approaches can mask critical performance issues. ExtremeWeatherBench emphasizes event and case-based metrics that evaluate predictions at the level of discrete weather events. This approach better aligns with how forecasts are actually used by decision-makers and stakeholders. By focusing on user-driven metrics that directly measure a model's ability to predict the timing, location, and intensity of specific extreme events, we provide evaluations that are more relevant to real-world applications such as emergency management, resource allocation, and public safety warnings. This user-centric approach ensures that model improvements translate to meaningful societal benefits alongside statistical improvements.

## About ExtremeWeatherBench

ExtremeWeatherBench provides a standardized framework for evaluating machine learning models on extreme weather prediction tasks. It includes curated datasets, evaluation metrics, and baseline models specifically designed for extreme weather phenomena such as:

- Heat waves
- Freeze events (cold snaps)
- Severe Weather Days
- Tropical cyclones
- Atmospheric rivers
- Extreme precipitation events


## Features

- **AIWP and Evaluation datasets**: Pre-processed AIWP forecasts ready for analysis and curated global point observations from NCEI's Integrated Surface Dataset (ISD) stored on Brightband's cloud storage
- **Standardized evaluation**: Consistent foundational metrics for fair model comparison
- **Easy to use**: Simple and intuitive API for researchers and practitioners, including a friendly learning curve to build bespoke metrics and integration of other forecast and observational datasets
- **Well documented**: Comprehensive documentation with examples

## Installation

You can install ExtremeWeatherBench using: 

```
pip install git+https://github.com/brightband/ExtremeWeatherBench.git
```

Alternatively, for developers, you can install the package in development mode:

``` 
git clone https://github.com/brightband/ExtremeWeatherBench.git
pip install -e .
```

.. toctree::
    api.md