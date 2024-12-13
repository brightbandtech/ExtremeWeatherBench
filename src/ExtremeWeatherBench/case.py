import dataclasses
import xarray as xr
import typing as t
from collections import namedtuple
from . import utils

class Case:
    """
    Case holds the event and climatology metadata for a given case.
    It also holds the metadata for the location and box length width in km. 
    Attributes:
        location_centroid: dict: the latitude and longitude of the center of the location
        box_length_width_in_km: int: the side length of the square in kilometers
        analysis_variables: t.Union[None, t.List[str]] = None: variable names for the analysis dataset, optional
        forecast_variables: t.Union[None, t.List[str]] = None: variable names for the forecast dataset, optional
    """

    def __init__(self):
        self.convert_longitude()
        self.process_data_to_spec()
        self.align_climatology_to_case_analysis_time()
        #TODO: include obs data calls here from ISD
        
    def convert_longitude(self):
        self.case_analysis_ds = utils.convert_longitude_to_180(self.case_analysis_ds)
        self.climatology_ds = utils.convert_longitude_to_180(self.climatology_ds)

    def process_data_to_spec(self):
        self.case_analysis_ds = self.clip_and_remove_ocean(self.case_analysis_ds)
        self.climatology_ds = self.clip_and_remove_ocean(self.climatology_ds)
    
    def align_climatology_to_case_analysis_time(self):
        self.climatology_ds = self.climatology_ds.sel(time=self.case_analysis_ds.time)
    
    def clip_and_remove_ocean(self, xarray_data_object: t.Union[xr.Dataset,xr.DataArray]):
        interim_xarray_data_object = utils.clip_dataset_to_square(xarray_data_object, self.location_centroid, self.box_length_width_in_km)
        output_xarray_data_object = utils.remove_ocean_gridpoints(interim_xarray_data_object)
        return output_xarray_data_object