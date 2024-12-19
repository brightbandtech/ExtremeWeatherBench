import xarray as xr
import typing as t
import utils
import numpy as np
import scores

#Intensity heat metrics
def threshold_weighted_rmse(da_fcst: xr.DataArray, da_obs: xr.DataArray, threshold: float, threshold_tolerance: float):
    mse = scores.continuous.tw_squared_error(da_fcst, 
                                             da_obs, 
                                             interval_where_one=(threshold, np.inf), 
                                             interval_where_positive=(threshold-threshold_tolerance, np.inf)
                                             )
    rmse = np.sqrt(mse)
    return rmse

def mae_max_of_max_temperatures(da_fcst: xr.DataArray, da_obs: xr.DataArray):
    mae = scores.continuous.mae(da_fcst.groupby('time.day').max().max(dim='time'),
                                da_obs.groupby('time.day').max().max(dim='time'))
    return mae

def mae_max_of_min_temperatures(da_fcst: xr.DataArray, da_obs: xr.DataArray):
    mae = scores.continuous.mae(da_fcst.groupby('time.day').min().max(dim='time'),
                                da_obs.groupby('time.day').min().max(dim='time'))
    return mae

#Duration heat metrics
def onset_above_85th_percentile(da_fcst: xr.DataArray, da_obs: xr.DataArray, da_clim_85th: xr.DataArray):
    def first_above_threshold(da, threshold):
        above_threshold = da > threshold
        first_time = above_threshold.argmin(dim='time')
        return first_time

    fcst_first_above = first_above_threshold(da_fcst, da_clim_85th)
    obs_first_above = first_above_threshold(da_obs, da_clim_85th)
    
    onset_me = (fcst_first_above - obs_first_above).astype(float)
    onset_me_hours = onset_me * np.timedelta64(1, 'h')
    return onset_me_hours

def mae_onset_and_end_above_85th_percentile(da_fcst: xr.DataArray, da_obs: xr.DataArray, da_clim_85th: xr.DataArray):
    def first_and_last_above_threshold(da, threshold):
        above_threshold = da > threshold
        first_time = above_threshold.argmax(dim='time')
        last_time = len(da['time']) - above_threshold[::-1].argmax(dim='time') - 1
        return first_time, last_time

    fcst_first_above, fcst_last_above = first_and_last_above_threshold(da_fcst, da_clim_85th)
    obs_first_above, obs_last_above = first_and_last_above_threshold(da_obs, da_clim_85th)
    
    onset_mae = np.abs(fcst_first_above - obs_first_above).astype(float)
    end_mae = np.abs(fcst_last_above - obs_last_above).astype(float)
    
    mean_absolute_error = ((onset_mae + end_mae) / 2) * np.timedelta64(1, 'h')
    return mean_absolute_error
