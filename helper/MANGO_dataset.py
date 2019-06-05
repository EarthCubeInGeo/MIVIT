# mango_dataset.py

from mangopy.mosaic import Mosaic
import datetime as dt
import numpy as np
from mivit import DataSet

def MANGO_dataset(targtime, plot_type=None, mangopy_kwargs=None):
    m = Mosaic(**mangopy_kwargs)
    mosaic, __, __, mosaic_lat, mosaic_lon, time = m.create_mosaic(targtime,cell_edges=True)
    # remove empty strings from list of times
    time = [t for t in time if t]
    time_range = [min(time), max(time)+dt.timedelta(minutes=5)]

    altitude = np.full(mosaic_lon.shape, 250.)
    # dataset = DataSet(longitude=mosaic_lon,latitude=mosaic_lat,values=mosaic,cmap='gist_gray',plot_type='pcolormesh', instrument='MANGO', parameter='Brightness')
    dataset = DataSet(longitude=mosaic_lon,latitude=mosaic_lat,altitude=altitude,values=mosaic,time_range=time_range, name='MANGO')
    return dataset
