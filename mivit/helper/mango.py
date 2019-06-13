# mango.py

import mangopy as mango
import datetime as dt
import numpy as np
from ..dataset import DataSet

def camera(targtime, site, mangopy_kwargs=None):
    m = mango.Mango(**mangopy_kwargs)
    s = m.get_site_info(site)

    img, lat, lon, time = m.get_data(s,targtime)
    time_range = [time, time+dt.timedelta(minutes=5)]
    alt = np.full(lat.shape, 250.)
    dataset = DataSet(longitude=lon, latitude=lat, altitude=alt, values=img, time_range=time_range, name='MANGO {}'.format(s['name']))
    return dataset

def mosaic(targtime, mangopy_kwargs=None):
    m = mango.Mosaic(**mangopy_kwargs)
    mosaic, __, __, lat, lon, time = m.create_mosaic(targtime,cell_edges=True)
    # remove empty strings from list of times
    time = [t for t in time if t]
    time_range = [min(time), max(time)+dt.timedelta(minutes=5)]

    alt = np.full(lat.shape, 250.)
    dataset = DataSet(longitude=lon, latitude=lat, altitude=alt, values=mosaic, time_range=time_range, name='MANGO Mosaic')
    return dataset
