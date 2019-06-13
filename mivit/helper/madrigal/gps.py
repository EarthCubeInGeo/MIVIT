# gps.py

from madrigal import identify_file, target_index
import numpy as np
import datetime as dt
import h5py
from ...dataset import DataSet

def tec(targtime, user_info):

    instrument_code = 8000
    file_code = 3500

    filename = identify_file(targtime,instrument_code,file_code, user_info)

    with h5py.File(filename,'r') as file:
        tstmp = file['/Data/Array Layout/timestamps'][:]
        i = target_index(targtime,tstmp)
        time_range = [dt.datetime.utcfromtimestamp(tstmp[i]),dt.datetime.utcfromtimestamp(tstmp[i+1])]
        latitude = file['/Data/Array Layout/gdlat'][:]
        longitude = file['/Data/Array Layout/glon'][:]
        tec = file['/Data/Array Layout/2D Parameters/tec'][:,:,i]
    Lon, Lat = np.meshgrid(longitude,latitude)
    Alt = np.full(Lon.shape, 350.)
    dataset = DataSet(values=tec,latitude=Lat,longitude=Lon,altitude=Alt, time_range=time_range, name='GPS TEC')
    return dataset

