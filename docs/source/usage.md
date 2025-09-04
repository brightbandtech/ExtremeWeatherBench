# A Quickstart to ExtremeWeatherBench

There are two main ways to use ExtremeWeatherBench, by script or by command line.

To run the Brightband-based evaluation on an existing AIWP model (FCN v2), which 
includes the default 290 cases for heat waves, freezes, severe convective days, 
tropical cyclones, and atmospheric rivers:

```bash
ewb --default
```

or:

```python
from extremeweatherbench import evaluate, defaults, utils

eval_objects = defaults.BRIGHTBAND_EVALUATION_OBJECTS

cases = utils.load_events_yaml()
ewb = ExtremeWeatherBench(cases=cases, 
evaluation_objects=eval_objects)

outputs = ewb.run()

outputs.to_csv('your_outputs.csv')
```