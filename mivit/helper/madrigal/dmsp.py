# dmsp.py

from madrigal import identify_file, target_index
import numpy as np
import datetime as dt
import h5py
from ...dataset import DataSet

def density(starttime, endtime, user_info, madrigal_dir=None):

    instrument_code = 8100
    file_code = 10245       # F15 with UT quality flags

    filename = identify_file(starttime,instrument_code,file_code, user_info, madrigal_dir)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx1 = target_index(starttime,tstmp)
        idx2 = target_index(endtime,tstmp)
        time_range = [dt.datetime.utcfromtimestamp(tstmp[idx1]), dt.datetime.utcfromtimestamp(tstmp[idx2])]
        dens = file['/Data/Table Layout']['ni'][idx1:idx2]
        lat = file['/Data/Table Layout']['gdlat'][idx1:idx2]
        lon = file['/Data/Table Layout']['glon'][idx1:idx2]
        alt = file['/Data/Table Layout']['gdalt'][idx1:idx2]

    dataset = DataSet(values=dens,latitude=lat,longitude=lon,altitude=alt, time_range=time_range, name='DMSP')
    return dataset


def velocity(starttime, endtime, user_info, madrigal_dir=None):
    instrument_code = 8100
    file_code = 10245       # F15 with UT quality flags

    filename = identify_file(starttime,instrument_code,file_code, user_info, madrigal_dir)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx1 = target_index(starttime,tstmp)
        idx2 = target_index(endtime,tstmp)
        time_range = [dt.datetime.utcfromtimestamp(tstmp[idx1]), dt.datetime.utcfromtimestamp(tstmp[idx2])]
        forw = file['/Data/Table Layout']['ion_v_sat_for'][idx1:idx2]
        left = file['/Data/Table Layout']['ion_v_sat_left'][idx1:idx2]
        vert = file['/Data/Table Layout']['vert_ion_v'][idx1:idx2]
        lat = file['/Data/Table Layout']['gdlat'][idx1:idx2]
        lon = file['/Data/Table Layout']['glon'][idx1:idx2]
        alt = file['/Data/Table Layout']['gdalt'][idx1:idx2]

    # set forward values to zero (RAM velocities are difficult to interpret)
    forw = np.zeros(forw.shape)
    dataset = DataSet(values=np.array([forw,left,vert]), latitude=lat, longitude=lon, altitude=alt, time_range=time_range, name='DMSP Velocity', sat_comp=True)
    return dataset


