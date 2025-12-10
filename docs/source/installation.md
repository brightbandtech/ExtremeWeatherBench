
## Using uv
[uv](https://docs.astral.sh/uv/) is strongly recommended to use when installing EWB.

You can install ExtremeWeatherBench using: 

```
uv pip install git+https://github.com/brightband/ExtremeWeatherBench.git
uv sync --all-extras
```

## Using pip
Alternatively, you can use pip in the same fashion (using a virtual environment is recommended):
```
pip install git+https://github.com/brightband/ExtremeWeatherBench.git
```

Alternatively, for developers, you can install the package in development mode:


``` 
git clone https://github.com/brightband/ExtremeWeatherBench.git
uv sync --all-extras
```