# superdarn.py

from davitpy import pydarn
import numpy as np
from ..dataset import DataSet

# TODO: These functions are all VERY similar - should probably be consolidated using some kind of more general function to avoid replicating code

def velocity(targtime, radar, davitpy_kwargs=None):

    sdptr = pydarn.sdio.radDataOpen(targtime,radar,**davitpy_kwargs)
    scan = sdptr.readScan()

    site = pydarn.radar.site(radId=scan[0].stid,dt=scan[0].time)
    fov = pydarn.radar.radFov.fov(site=site,rsep=scan[0].prm.rsep,ngates=scan[0].prm.nrang+1,nbeams=site.maxbeam,coords='geo',date_time=scan[0].time)

    velocity = np.full(fov.latCenter.shape,np.nan)
    for beam in scan:
        for k, r in enumerate(beam.fit.slist):
            velocity[beam.bmnum,r] = beam.fit.v[k]

    # get range of times scan covers
    time_range = [scan[0].time,scan[-1].time]

    altitude = np.full(np.array(fov.lonFull).shape, 400.)

    dataset = DataSet(longitude=np.array(fov.lonFull),latitude=np.array(fov.latFull),altitude=altitude,values=velocity,time_range=time_range,name='SuperDARN {}'.format(radar.upper()))

    return dataset

def power(targtime, radar, davitpy_kwargs=None):

    sdptr = pydarn.sdio.radDataOpen(targtime,radar,**davitpy_kwargs)
    scan = sdptr.readScan()

    site = pydarn.radar.site(radId=scan[0].stid,dt=scan[0].time)
    fov = pydarn.radar.radFov.fov(site=site,rsep=scan[0].prm.rsep,ngates=scan[0].prm.nrang+1,nbeams=site.maxbeam,coords='geo',date_time=scan[0].time)

    power = np.full(fov.latCenter.shape,np.nan)
    for beam in scan:
        for k, r in enumerate(beam.fit.slist):
            power[beam.bmnum,r] = beam.fit.p_l[k]

    # get range of times scan covers
    time_range = [scan[0].time,scan[-1].time]

    altitude = np.full(np.array(fov.lonFull).shape, 400.)

    dataset = DataSet(longitude=np.array(fov.lonFull),latitude=np.array(fov.latFull),altitude=altitude,values=power,time_range=time_range,name='SuperDARN {}'.format(radar.upper()))

    return dataset

def spectralwidth(targtime, radar, davitpy_kwargs=None):

    sdptr = pydarn.sdio.radDataOpen(targtime,radar,**davitpy_kwargs)
    scan = sdptr.readScan()

    site = pydarn.radar.site(radId=scan[0].stid,dt=scan[0].time)
    fov = pydarn.radar.radFov.fov(site=site,rsep=scan[0].prm.rsep,ngates=scan[0].prm.nrang+1,nbeams=site.maxbeam,coords='geo',date_time=scan[0].time)

    spectralwidth = np.full(fov.latCenter.shape,np.nan)
    for beam in scan:
        for k, r in enumerate(beam.fit.slist):
            spectralwidth[beam.bmnum,r] = beam.fit.w_l[k]

    # get range of times scan covers
    time_range = [scan[0].time,scan[-1].time]

    altitude = np.full(np.array(fov.lonFull).shape, 400.)

    dataset = DataSet(longitude=np.array(fov.lonFull),latitude=np.array(fov.latFull),altitude=altitude,values=spectralwidth,time_range=time_range,name='SuperDARN {}'.format(radar.upper()))

    return dataset