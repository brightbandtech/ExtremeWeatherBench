
## Using uv
[uv](https://docs.astral.sh/uv/) is strongly recommended to use when installing EWB.

You can install ExtremeWeatherBench using: 

```
uv add extremeweatherbench
```

## Using pip
Alternatively, you can use pip in the same fashion (using a virtual environment is recommended):
```
pip install extremeweatherbench
```

For development, an easy approach is clone the repository and use `uv` to sync all required and optional dependencies:

``` 
git clone https://github.com/brightbandtech/ExtremeWeatherBench.git
uv sync --all-extras
```