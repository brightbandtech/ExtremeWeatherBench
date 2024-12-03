"""
Utility functions for other files that can apply to multiple files in the library
"""

import numpy as np
import pandas as pd
import ujson
from kerchunk.hdf import SingleHdf5ToZarr 

def convert_longitude_to_360(longitude):
    return longitude % 360

def generate_json_from_nc(u,so,fs,fs_out,json_dir):
    with fs.open(u, **so) as infile:
        h5chunks = SingleHdf5ToZarr(infile, u, inline_threshold=300)

        file_split = u.split('/') # seperate file path to create a unique name for each json 
        model = file_split[1].split('_')[0]
        date_string = file_split[-1].split('_')[3]
        outf = f'{json_dir}{model}_{date_string}_.json'
        print(outf)
        with fs_out.open(outf, 'wb') as f:
            f.write(ujson.dumps(h5chunks.translate()).encode());

# seasonal aggregation functions for max, min, and mean
def seasonal_subset_max(df):
    df = df.where(df.index.month.isin([6,7,8]))
    return df.max()

def seasonal_subset_min(df):
    df = df.where(df.index.month.isin([6,7,8]))
    return df.min()

def seasonal_subset_mean(df):
    df = df.where(df.index.month.isin([6,7,8]))
    return df.mean()

def is_jja(month):
    return (month >= 6) & (month <= 8)

def is_6_hourly(hour):
    return (hour == 0) | (hour == 6) | (hour == 12) | (hour == 18)