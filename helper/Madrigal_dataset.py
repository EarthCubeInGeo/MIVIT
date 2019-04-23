# Madrigal_dataset.py

import madrigalWeb.madrigalWeb
import datetime as dt
import numpy as np
import h5py
import os
from mivit import DataSet

def GPSTEC_dataset(targtime, user_info):

    instrument_code = 8000
    file_code = 3500

    filename = identify_file(targtime,instrument_code,file_code, user_info)

    with h5py.File(filename,'r') as file:
        tstmp = file['/Data/Array Layout/timestamps'][:]
        i = target_index(targtime,tstmp)
        latitude = file['/Data/Array Layout/gdlat'][:]
        longitude = file['/Data/Array Layout/glon'][:]
        tec = file['/Data/Array Layout/2D Parameters/tec'][:,:,i]
    Lon, Lat = np.meshgrid(longitude,latitude)
    dataset = DataSet(values=tec,latitude=Lat,longitude=Lon,cmap='terrain',plot_type='contourf', instrument='GPS', parameter='TEC', plot_kwargs={'alpha':0.2, 'levels':25})
    return dataset

def MLHISR_dataset(targtime, user_info):

    instrument_code = 30
    file_code = 3430

    filename = identify_file(targtime,instrument_code,file_code, user_info)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /timestamps'][:]
        idx = target_index(targtime,tstmp)
        rangegate = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /range'][:]
        azimuth = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /1D Parameters/az1'][idx]
        elevation = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /1D Parameters/el1'][idx]
        density = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /2D Parameters/ne'][:,idx]

        site = file['/Metadata/Experiment Parameters']['value'][8:11]
        site = [float(s) for s in site]

    dataset = DataSet(values=density, site=site, azimuth=azimuth, elevation=elevation, ranges=rangegate, cmap='jet', instrument='Millstone Hill ISR', parameter='Ne')
    return dataset

def MLHFPI_dataset(targtime, user_info):

    instrument_code = 5360
    file_code = 7100

    filename = identify_file(targtime,instrument_code,file_code, user_info)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx = target_index(targtime,tstmp)
        Tn = file['/Data/Table Layout']['tn'][idx-2:idx+3]
        azimuth = file['/Data/Table Layout']['azm'][idx-2:idx+3]
        elevation = file['/Data/Table Layout']['elm'][idx-2:idx+3]
        altitude = file['/Data/Table Layout']['alte'][idx-2:idx+3]
        site = file['/Metadata/Experiment Parameters']['value'][8:11]
        site = [float(s) for s in site]
    # times = np.array([dt.datetime.utcfromtimestamp(t) for t in tstmp])
    dataset = DataSet(values=Tn, site=site, azimuth=azimuth, elevation=elevation, altitude=altitude, cmap='cool', instrument='Millstone Hill FPI', parameter='Tn')
    return dataset

def MLHFPIvec_dataset(targtime, user_info):

    instrument_code = 5360
    file_code = 7101

    filename = identify_file(targtime,instrument_code,file_code, user_info)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx = target_index(targtime,tstmp)
        ve = file['/Data/Table Layout']['vn1'][idx]
        vn = file['/Data/Table Layout']['vn2'][idx+1]
        latitude = file['/Data/Table Layout']['gdlat'][idx]
        longitude = file['/Data/Table Layout']['glon'][idx]
        altitude = file['/Data/Table Layout']['alte'][idx]
    # time = dt.datetime.utcfromtimestamp(tstmp)
    dataset = DataSet(values=np.array([np.array([ve]),np.array([vn]),np.array([0.])]), latitude=np.array([latitude]), longitude=np.array([longitude]), altitude=np.array([altitude]), plot_type='quiver', instrument='Millstone Hill FPI', parameter='Vn')
    return dataset

def identify_file(t,instrument_code,file_code, user):

    # initialize MadrigalData object and establish conection with Madrigal website
    test =  madrigalWeb.madrigalWeb.MadrigalData('http://cedar.openmadrigal.org/')

    # find experiments on day in question
    expList = test.getExperiments(instrument_code, t.year, t.month, t.day, 0, 0, 1, t.year, t.month, t.day, 23, 59, 59)
    # find files associated with experiment
    fileList = test.getExperimentFiles(expList[0].id)
    # find files of the correct type (correct file code)
    datafile = [file.name for file in fileList if file.kindat==file_code][0]

    # create file path by appending the Madarigal file name to the data directory
    filename = './TestDataSets/'+datafile.split('/')[-1]

    # if file does not exist, download it
    if not os.path.exists(filename):
        print('DOWNLOADING '+filename)
        test.downloadFile(datafile,filename, user['fullname'], user['email'], user['affiliation'], 'hdf5')

    return filename


def target_index(targtime,tstmparray):
    # convert targtime to unix timestamp
    targtstmp = (targtime-dt.datetime.utcfromtimestamp(0)).total_seconds()
    # find index of time in timearray that is closest to targtime
    t = np.argmin(np.abs(tstmparray-targtstmp))
    return t
