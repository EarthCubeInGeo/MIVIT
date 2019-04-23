# SuperDARN_dataset.py

from davitpy import pydarn
import davitpy.pydarn.sdio
import numpy as np
from mivit import DataSet

def SuperDARN_dataset(targtime,radar, davitpy_kwargs=None):

    sdptr = pydarn.sdio.radDataOpen(targtime,radar,**davitpy_kwargs)
    scan = sdptr.readScan()

    site = pydarn.radar.site(radId=scan[0].stid,dt=scan[0].time)
    fov = pydarn.radar.radFov.fov(site=site,rsep=scan[0].prm.rsep,ngates=scan[0].prm.nrang+1,nbeams=site.maxbeam,coords='geo',date_time=scan[0].time)

    velocity = np.full(fov.latCenter.shape,np.nan)
    for beam in scan:
        for k, r in enumerate(beam.fit.slist):
            velocity[beam.bmnum,r] = beam.fit.v[k]

    dataset = DataSet(longitude=np.array(fov.lonFull),latitude=np.array(fov.latFull),values=np.array(velocity),cmap='bwr',plot_type='pcolormesh',instrument='SuperDARN '+radar.upper(), parameter='Velocity',plot_kwargs={'vmin':-40,'vmax':40})
    return dataset