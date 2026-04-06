"""ExtremeWeatherBench: A benchmarking framework for extreme weather forecasts.

    import extremeweatherbench as ewb

    ewb.evaluate.ExtremeWeatherBench(case_metadata=..., evaluation_objects=...)
    ewb.inputs.ERA5(...)
    ewb.inputs.ZarrForecast(...)
    ewb.metrics.MeanAbsoluteError(...)
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "calc",
        "cases",
        "defaults",
        "derived",
        "evaluate",
        "inputs",
        "metrics",
        "regions",
        "utils",
    ],
)
