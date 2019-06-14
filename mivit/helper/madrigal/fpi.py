# fpi.py

from madrigal import identify_file, target_index
import numpy as np
import datetime as dt
import h5py
from ...dataset import DataSet

def Tn(targtime, line, user_info, madrigal_dir=None):

    file_codes = {'red':7100,'green':7110}

    instrument_code = 5360
    file_code = file_codes[line]

    filename = identify_file(targtime,instrument_code,file_code, user_info, madrigal_dir)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx = target_index(targtime,tstmp)
        stime = file['/Data/Table Layout']['ut1_unix'][idx-2]
        etime = file['/Data/Table Layout']['ut2_unix'][idx+3]
        time_range = [dt.datetime.utcfromtimestamp(stime),dt.datetime.utcfromtimestamp(etime)]
        Tn = file['/Data/Table Layout']['tn'][idx-2:idx+3]
        azimuth = file['/Data/Table Layout']['azm'][idx-2:idx+3]
        elevation = file['/Data/Table Layout']['elm'][idx-2:idx+3]
        altitude = file['/Data/Table Layout']['alte'][idx-2:idx+3]
        site = file['/Metadata/Experiment Parameters']['value'][8:11]
        site = [float(s) for s in site]
    dataset = DataSet(values=Tn, site=site, azimuth=azimuth, elevation=elevation, altitude=altitude, time_range=time_range, name='Neutral Temperature')
    return dataset

def Vn(targtime, line, user_info, madrigal_dir=None):

    file_codes = {'red':7101,'green':7111}

    instrument_code = 5360
    file_code = file_codes[line]

    filename = identify_file(targtime,instrument_code,file_code, user_info, madrigal_dir)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx = target_index(targtime,tstmp)
        stime = file['/Data/Table Layout']['ut1_unix'][idx]
        etime = file['/Data/Table Layout']['ut2_unix'][idx+1]
        time_range = [dt.datetime.utcfromtimestamp(stime),dt.datetime.utcfromtimestamp(etime)]
        ve = file['/Data/Table Layout']['vn1'][idx]
        vn = file['/Data/Table Layout']['vn2'][idx+1]
        latitude = file['/Data/Table Layout']['gdlat'][idx]
        longitude = file['/Data/Table Layout']['glon'][idx]
        altitude = file['/Data/Table Layout']['alte'][idx]
    dataset = DataSet(values=np.array([np.array([ve]),np.array([vn]),np.array([0.])]), latitude=np.array([latitude]), longitude=np.array([longitude]), altitude=np.array([altitude]), time_range=time_range, name='Neutral Wind Velocity')
    return dataset
