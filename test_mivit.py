# test_mivit.py


import numpy as np
import datetime as dt
import h5py
import cartopy.crs as ccrs
from mangopy.mosaic import Mosaic
from davitpy import pydarn
import davitpy.pydarn.sdio
from mivit import DataSet, Visualize



def test():


    # get SuperDARN data
    sdtime = dt.datetime(2016,10,1,17,0)
    sd_data = []
    for rad in ['sas','kap','pgr']:
        sdptr = pydarn.sdio.radDataOpen(sdtime,rad,src='local',fileType='fitex',local_dirfmt='./TestDataSets/SuperDARN/')
        scan = sdptr.readScan()

        site = pydarn.radar.site(radId=scan[0].stid,dt=scan[0].time)
        fov = pydarn.radar.radFov.fov(site=site,rsep=scan[0].prm.rsep,ngates=scan[0].prm.nrang+1,nbeams=site.maxbeam,coords='geo',date_time=scan[0].time)

        velocity = np.full(fov.latCenter.shape,np.nan)
        for beam in scan:
            for k, r in enumerate(beam.fit.slist):
                velocity[beam.bmnum,r] = beam.fit.v[k]
        # sd_data.append(DataSet(longitude=np.array(fov.lonCenter),latitude=np.array(fov.latCenter),values=np.array(velocity),cmap='bwr',instrument='SuperDARN '+rad.upper(), parameter='Velocity'))
        sd_data.append(DataSet(longitude=np.array(fov.lonFull),latitude=np.array(fov.latFull),values=np.array(velocity),cmap='bwr',plot_type='pcolormesh',instrument='SuperDARN '+rad.upper(), parameter='Velocity'))



    # get mango data
    targtime = dt.datetime(2017,5,28,5,35)
    m = Mosaic()
    mosaic, mosaic_lat, mosaic_lon = m.create_mosaic(targtime)
    mango = DataSet(longitude=mosaic_lon,latitude=mosaic_lat,values=mosaic,cmap='gist_gray',plot_type='pcolormesh', instrument='MANGO', parameter='Brightness')



    # get GPS TEC
    filename = './TestDataSets/gps170528g.004.hdf5'
    with h5py.File(filename,'r') as file:
        tstmp = file['/Data/Array Layout/timestamps'][:]
        i = target_index(targtime,tstmp)
        latitude = file['/Data/Array Layout/gdlat'][:]
        longitude = file['/Data/Array Layout/glon'][:]
        tec = file['/Data/Array Layout/2D Parameters/tec'][:,:,i]
    Lon, Lat = np.meshgrid(longitude,latitude)
    tec = DataSet(values=tec,latitude=Lat,longitude=Lon,cmap='jet',plot_type='contour', instrument='GPS', parameter='TEC')



    # get Millstone Hill data
    filename = './TestDataSets/mlh170608k.004.hdf5'
    with h5py.File(filename, 'r') as file:
        idx = 36
        tstmp = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /timestamps'][idx]
        rangegate = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /range'][:]
        azimuth = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /1D Parameters/az1'][idx]
        elevation = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /1D Parameters/el1'][idx]
        density = file['/Data/Array Layout/Array with kinst=31.0 and mdtyp=115.0 and pl=0.00048 /2D Parameters/ne'][:,idx]
    mlh = DataSet(values=density, site=[42.62,-71.49,0.0], azimuth=azimuth, elevation=elevation, ranges=rangegate, cmap='jet', instrument='Millstone Hill ISR', parameter='Ne')


    # get Millston Hill FPI data
    filename = './TestDataSets/kfp170527g.7110.hdf5'
    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][60:65]
        Tn = file['/Data/Table Layout']['tn'][60:65]
        azimuth = file['/Data/Table Layout']['azm'][60:65]
        elevation = file['/Data/Table Layout']['elm'][60:65]
        altitude = file['/Data/Table Layout']['alte'][60:65]

    times = np.array([dt.datetime.utcfromtimestamp(t) for t in tstmp])
    mlh_fpi = DataSet(values=Tn, site=[42.62,-71.49,0.0], azimuth=azimuth, elevation=elevation, altitude=altitude, cmap='cool', instrument='Millstone Hill FPI', parameter='Tn')


    # get Millston Hill vector FPI data
    filename = './TestDataSets/kfp170527g.7111.hdf5'
    with h5py.File(filename, 'r') as file:
        tstmp = file['/Data/Table Layout']['ut1_unix'][50]
        ve = file['/Data/Table Layout']['vn1'][50]
        vn = file['/Data/Table Layout']['vn2'][49]
        latitude = file['/Data/Table Layout']['gdlat'][50]
        longitude = file['/Data/Table Layout']['glon'][50]
        altitude = file['/Data/Table Layout']['alte'][50]
    time = dt.datetime.utcfromtimestamp(tstmp)
    mlh_fpi_vec = DataSet(values=np.array([np.array([ve]),np.array([vn]),np.array([0.])]), latitude=np.array([latitude]), longitude=np.array([longitude]), altitude=np.array([altitude]), cmap='jet', plot_type='quiver', instrument='Millstone Hill FPI', parameter='Vn')





    plot = Visualize([mango,tec,mlh,mlh_fpi,mlh_fpi_vec]+sd_data)
    plot.mlat_mlon=True
    plot.one_map()
    # plot.multi_map()



def target_index(targtime,tstmparray):
    # convert targtime to unix timestamp
    targtstmp = (targtime-dt.datetime.utcfromtimestamp(0)).total_seconds()
    # find index of time in timearray that is closest to targtime
    t = np.argmin(np.abs(tstmparray-targtstmp))
    return t



def main():
    test()

if __name__ == '__main__':
    main()