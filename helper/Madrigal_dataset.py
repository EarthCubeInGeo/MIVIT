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
        time_range = [dt.datetime.utcfromtimestamp(tstmp[i]),dt.datetime.utcfromtimestamp(tstmp[i+1])]
        latitude = file['/Data/Array Layout/gdlat'][:]
        longitude = file['/Data/Array Layout/glon'][:]
        tec = file['/Data/Array Layout/2D Parameters/tec'][:,:,i]
    Lon, Lat = np.meshgrid(longitude,latitude)
    # dataset = DataSet(values=tec,latitude=Lat,longitude=Lon,cmap='magma',plot_type='contourf', instrument='GPS', parameter='TEC', plot_kwargs={'alpha':0.2, 'levels':25})
    dataset = DataSet(values=tec,latitude=Lat,longitude=Lon, time_range=time_range, name='GPS TEC')
    return dataset

def DMSP_dataset(targtime, user_info):

    instrument_code = 8100
    file_code = 10245       # F15 with UT quality flags

    filename = identify_file(targtime,instrument_code,file_code, user_info)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx1 = target_index(targtime-dt.timedelta(hours=1),tstmp)
        idx2 = target_index(targtime+dt.timedelta(hours=1),tstmp)
        time_range = [dt.datetime.utcfromtimestamp(tstmp[idx1]), dt.datetime.utcfromtimestamp(tstmp[idx2])]
        dens = file['/Data/Table Layout']['ni'][idx1:idx2]
        lat = file['/Data/Table Layout']['gdlat'][idx1:idx2]
        lon = file['/Data/Table Layout']['glon'][idx1:idx2]
        alt = file['/Data/Table Layout']['gdalt'][idx1:idx2]

    dataset = DataSet(values=dens,latitude=lat,longitude=lon,altitude=alt, time_range=time_range, name='DMSP')
    return dataset


def DMSPvec_dataset(targtime, user_info):
    instrument_code = 8100
    file_code = 10245       # F15 with UT quality flags

    filename = identify_file(targtime,instrument_code,file_code, user_info)

    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][:]
        idx1 = target_index(targtime-dt.timedelta(hours=1),tstmp)
        idx2 = target_index(targtime+dt.timedelta(hours=1),tstmp)
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


def PFISR_dataset(targtime, user_info):

    instrument_code = 61
    file_code = 5950

    filename = identify_file(targtime,instrument_code,file_code, user_info)
    print filename

    rangegate = []
    azimuth = []
    elevation = []
    density = []

    # import visuamisr
    # data = visuamisr.read_data(filename)
    with h5py.File(filename, 'r') as file:
        beams = file['Data/Array Layout']
        for b in beams.keys():
            tstmp = beams[b]['timestamps'][:]
            idx = target_index(targtime,tstmp)
            print idx

            r = beams[b]['range'][:]
            print r, r.shape
            az = beams[b]['1D Parameters/azm'][idx]
            el = beams[b]['1D Parameters/elm'][idx]
            d = beams[b]['2D Parameters/nel'][:,idx]
            print type(r)

            rangegate.append(r)
            azimuth.append(az)
            elevation.append(el)
            density.append(d)

        site = file['/Metadata/Experiment Parameters']['value'][8:11]
        site = [float(s) for s in site]

    rangegate = np.stack(rangegate)
    # azimuth = np.array(azimuth)
    # elevation = np.array(elevation)
    # density = np.array(density)

    # print rangegate.shape, azimuth.shape, elevation.shape, density.shape
    print rangegate, rangegate.shape


    dataset = DataSet(values=density, site=site, azimuth=azimuth, elevation=elevation, ranges=rangegate, cmap='jet', instrument='PFISR', parameter='Ne')
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

def MLHFPI_dataset(targtime, line, user_info):

    file_codes = {'red':7100,'green':7110}

    instrument_code = 5360
    file_code = file_codes[line]

    filename = identify_file(targtime,instrument_code,file_code, user_info)

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
    # times = np.array([dt.datetime.utcfromtimestamp(t) for t in tstmp])
    # dataset = DataSet(values=Tn, site=site, azimuth=azimuth, elevation=elevation, altitude=altitude, cmap='cool', instrument='Millstone Hill FPI', parameter='Tn')
    dataset = DataSet(values=Tn, site=site, azimuth=azimuth, elevation=elevation, altitude=altitude, time_range=time_range, name='Neutral Temperature')
    return dataset

def MLHFPIvec_dataset(targtime, line, user_info):

    file_codes = {'red':7101,'green':7111}

    instrument_code = 5360
    file_code = file_codes[line]

    filename = identify_file(targtime,instrument_code,file_code, user_info)

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
    # time = dt.datetime.utcfromtimestamp(tstmp)
    # dataset = DataSet(values=np.array([np.array([ve]),np.array([vn]),np.array([0.])]), latitude=np.array([latitude]), longitude=np.array([longitude]), altitude=np.array([altitude]), plot_type='quiver', instrument='Millstone Hill FPI', parameter='Vn', plot_kwargs={'width':0.002})
    dataset = DataSet(values=np.array([np.array([ve]),np.array([vn]),np.array([0.])]), latitude=np.array([latitude]), longitude=np.array([longitude]), altitude=np.array([altitude]), time_range=time_range, name='Neutral Wind Velocity')
    return dataset

def identify_file(t,instrument_code,file_code, user):

    # initialize MadrigalData object and establish conection with Madrigal website
    test =  madrigalWeb.madrigalWeb.MadrigalData('http://cedar.openmadrigal.org/')

    # find experiments on day in question
    expList = test.getExperiments(instrument_code, t.year, t.month, t.day, 0, 0, 0, t.year, t.month, t.day, 23, 59, 59)
    ID = [exp.id for exp in expList if t>=dt.datetime(exp.startyear,exp.startmonth,exp.startday,exp.starthour,exp.startmin,exp.startsec) and t<dt.datetime(exp.endyear,exp.endmonth,exp.endday,exp.endhour,exp.endmin,exp.endsec)][0]
    # find files associated with experiment
    fileList = test.getExperimentFiles(ID)
    # print ID, fileList
    # for file in fileList:
    #     print file.kindat
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
