import xarray as xr
import pandas as pd
import numpy as np
import dataclasses

@dataclasses.dataclass
class Analysis:
    """
    This class is a parent class meant to be inherited for both ERA5
    and ISD analysis classes. It contains common functions 
    and initializations that are used in both classes.
    """
    data: xr.Dataset
    data_path: str
    start_date: str
    end_date: str
    location: dict
    location_center: dict
    length_km: float
    data_type: str
    data_name: str
    data_units: str
    data_description: str
    data_variable: str
    data_variable_units: str

    def __post_init__(self):
        self.data = self.data.sel(time=slice(self.start_date, self.end_date))
        self.data = self.data.rio.clip_box(**self.location)
        self.data = self.data.rio.clip_box(**self.location_center, length=self.length_km)
        self.data = self.data.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        self.data = self.data.rio.write_crs("epsg:4326", inplace=True)
        self.data = self.data.rename(longitude="lon", latitude="lat")
        self.data = self.data.drop_vars(["crs"])
        self.data = self.data.drop_dims(["spatial_ref"])
        self.data = self.data.drop_dims(["spatial_ref"])
        self.data = self.data.rename({self.data_variable: self.data_name})
        self.data.attrs["units"] = self.data_units
        self.data.attrs["description"] = self.data_description
        self.data.attrs["variable"] = self.data_variable
        self.data.attrs["variable_units"] = self.data_variable_units
        self.data.attrs["data_type"] = self.data_type

    def get_data(self):
        return self.data

    def get_data_path(self):
        return self.data_path

    def get_data_name(self):
        return self.data_name

    def get_data_units(self):
        return self.data_units

    def get_data_description(self):
        return self.data_description

    def get_data_variable(self):
        return self.data_variable

    def get_data_variable_units(self):
        return self.data_variable_units

    def get_data_type(self):
        return self.data_type

    def get_start_date(self):
        return self.start_date

    def get_end_date(self):
        return self.end_date

    def get_location(self):
        return self.location

    def get_location_center(self):
        return self.location_center

    def get_length_km(self):
        return self.length_km