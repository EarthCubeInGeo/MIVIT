# amisr.py

import visuamisr
import numpy as np
from ..dataset import DataSet

def density(targtime, filename):

    data = visuamisr.read_data(filename)

    times = data['ave_times']
    idx = np.argmin(np.abs([(t-targtime).total_seconds() for t in times]))

    dens = data['density'][idx,:,:]
    lat = data['latitude']
    lon = data['longitude']
    alt = data['altitude']
    time_range = data['times'][idx,:]

    dataset = DataSet(longitude=lon,latitude=lat,altitude=alt,values=dens, time_range=time_range, name='PFISR electron density')

    return dataset

