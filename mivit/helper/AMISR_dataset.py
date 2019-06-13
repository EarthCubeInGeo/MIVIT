# AMISR_dataset.py

import visuamisr
import numpy as np
from mivit import DataSet

def AMISR_dataset(targtime):

    filename = 'TestDataSets/20151107.003_lp_3min-fitcal.h5'    # hard-code file name for now
    data = visuamisr.read_data(filename)

    times = data['ave_times']
    idx = np.argmin(np.abs([(t-targtime).total_seconds() for t in times]))

    dens = data['density'][idx,:,:]

    lat = data['latitude']
    lon = data['longitude']
    alt = data['altitude']

    dataset = DataSet(longitude=lon,latitude=lat,altitude=alt,values=dens,cmap='jet',plot_type='scatter', instrument='PFISR', parameter='Density', plot_kwargs={'s':10})

    return dataset

