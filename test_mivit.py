# test_mivit.py


import numpy as np
import datetime as dt
import ConfigParser as configparser
import mivit
import copy


def test():

    targtime = dt.datetime(2017,5,28,6,35)

    # get SuperDARN data
    pt = mivit.PlotMethod(cmap='seismic',plot_type='pcolormesh',label='SuperDARN Velocity',vmin=-40,vmax=40)
    sd_data = []
    davitpy_kwargs = {'src':'local','fileType':'fitex','local_dirfmt':'./TestDataSets/SuperDARN/'}
    for rad in ['bks','fhe','fhw','kap','pgr','sas','wal']:
        sd = mivit.helper.superdarn.velocity(targtime,rad,davitpy_kwargs=davitpy_kwargs)
        sd_data.append(mivit.DataVisualization(sd, pt))


    # get mango data
    pt = mivit.PlotMethod(cmap='gist_gray',plot_type='pcolormesh', label='MANGO', vmin=0, vmax=255)
    mangopy_kwargs = {'datadir':'./TestDataSets/MANGO'}
    mango_data = mivit.helper.mango.mosaic(targtime,mangopy_kwargs=mangopy_kwargs)
    mango = mivit.DataVisualization(mango_data, pt)



    # set up madrigalWeb credentials
    config = configparser.ConfigParser()
    config.read('config.ini')
    user_fullname = config.get('DEFAULT', 'MADRIGAL_FULLNAME')
    user_email = config.get('DEFAULT', 'MADRIGAL_EMAIL')
    user_affiliation = config.get('DEFAULT', 'MADRIGAL_AFFILIATION')
    user_info = {'fullname':user_fullname,'email':user_email,'affiliation':user_affiliation}
    madrigal_dir = './TestDataSets'


    # get GPS TEC
    pt = mivit.PlotMethod(cmap='magma',plot_type='contourf', label='GPS TEC', alpha=0.2, levels=25, vmin=0, vmax=20)
    pt2 = mivit.PlotMethod(cmap='magma',plot_type='contour', label='GPS TEC', levels=25, vmin=0, vmax=20)
    tec_dat = mivit.helper.madrigal.gps.tec(targtime,user_info, madrigal_dir=madrigal_dir)
    tec = mivit.DataVisualization(tec_dat,[pt,pt2])


    # get Millston Hill FPI data
    fpi_g_dat = mivit.helper.madrigal.fpi.Tn(targtime, 'green', user_info, madrigal_dir=madrigal_dir)
    pt = mivit.PlotMethod(cmap='Greens', plot_type='scatter', label='FPI Tn', vmin=min(fpi_g_dat.values), vmax=max(fpi_g_dat.values), s=100)
    fpi_g = mivit.DataVisualization(fpi_g_dat, pt)

    pt = mivit.PlotMethod(color='green', plot_type='quiver', label='FPI Vn', width=0.002)
    fpi_vec_g_dat = mivit.helper.madrigal.fpi.Vn(targtime, 'green', user_info, madrigal_dir=madrigal_dir)
    fpi_vec_g = mivit.DataVisualization(fpi_vec_g_dat, pt)

    fpi_r_dat = mivit.helper.madrigal.fpi.Tn(targtime, 'red', user_info, madrigal_dir=madrigal_dir)
    pt = mivit.PlotMethod(cmap='Reds', plot_type='scatter', label='FPI Tn', vmin=min(fpi_r_dat.values), vmax=max(fpi_r_dat.values), s=30)    
    fpi_r = mivit.DataVisualization(fpi_r_dat, pt)

    pt = mivit.PlotMethod(color='red', plot_type='quiver', label='FPI Vn', width=0.002)
    fpi_vec_r_dat = mivit.helper.madrigal.fpi.Vn(targtime, 'red', user_info, madrigal_dir=madrigal_dir)
    fpi_vec_r = mivit.DataVisualization(fpi_vec_r_dat, pt)



    # get DMSP data
    pt = mivit.PlotMethod(cmap='jet',plot_type='scatter',label='DMSP Ni', vmin=0, vmax=3e10, s=20)
    dmsp_dat = mivit.helper.madrigal.dmsp.density(targtime-dt.timedelta(hours=1), targtime+dt.timedelta(hours=1), user_info, madrigal_dir=madrigal_dir)
    dmsp = mivit.DataVisualization(dmsp_dat, pt)

    pt = mivit.PlotMethod(cmap='jet',plot_type='quiver',label='DMSP Vi', width=0.002)
    dmsp_dat = mivit.helper.madrigal.dmsp.velocity(targtime-dt.timedelta(hours=1), targtime+dt.timedelta(hours=1), user_info, madrigal_dir=madrigal_dir)
    dmsp_vec = mivit.DataVisualization(dmsp_dat, pt)


    plot = mivit.Visualize([mango,tec,fpi_g,fpi_r,fpi_vec_g,fpi_vec_r,dmsp,dmsp_vec]+sd_data, map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    # plot = Visualize([dmsp,dmsp_vec], map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    plot.one_map()
    # plot.multi_map()


def test_mango():

    targtime = dt.datetime(2017,5,28,6,35)

    # get mango data
    mangopy_kwargs = {'datadir':'./TestDataSets/MANGO'}

    dataset = mivit.helper.mango.camera(targtime,'Hat Creek Observatory', mangopy_kwargs=mangopy_kwargs)
    pt = mivit.PlotMethod(cmap='jet',plot_type='scatter', label='MANGO', vmin=0, vmax=255, alpha=0.5, zorder=6)
    mango = mivit.DataVisualization(dataset, pt)

    dataset = mivit.helper.mango.mosaic(targtime, mangopy_kwargs=mangopy_kwargs)
    pt = mivit.PlotMethod(cmap='gist_gray',plot_type='pcolormesh', label='MANGO', vmin=0, vmax=255)
    mosaic = mivit.DataVisualization(dataset, pt)

    plot = mivit.Visualize([mango, mosaic], map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    plot.one_map()


def test_superdarn():

    targtime = dt.datetime(2017,5,28,6,35)

    ptv = mivit.PlotMethod(cmap='seismic',plot_type='pcolormesh',label='SuperDARN Velocity',vmin=-200,vmax=200)
    ptp = mivit.PlotMethod(cmap='Greens',plot_type='pcolormesh',label='SuperDARN Velocity',vmin=0,vmax=20)
    pts = mivit.PlotMethod(cmap='Oranges',plot_type='pcolormesh',label='SuperDARN Velocity',vmin=0,vmax=100)

    sd_data = []

    davitpy_kwargs = {'src':'local','fileType':'fitex','local_dirfmt':'./TestDataSets/SuperDARN/'}
    for rad in ['bks','fhe','fhw','kap','pgr','sas','wal']:

        # v = mivit.helper.superdarn.velocity(targtime,rad,davitpy_kwargs=davitpy_kwargs)
        # sd_data.append(mivit.DataVisualization(v, ptv))

        # p = mivit.helper.superdarn.power(targtime,rad,davitpy_kwargs=davitpy_kwargs)
        # sd_data.append(mivit.DataVisualization(p, ptp))

        s = mivit.helper.superdarn.spectralwidth(targtime,rad,davitpy_kwargs=davitpy_kwargs)
        sd_data.append(mivit.DataVisualization(s, pts))

    plot = mivit.Visualize(sd_data, map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    plot.one_map()

def test_amisr():

    targtime = dt.datetime(2017,5,28,6,35)

    filename = 'TestDataSets/20151107.003_lp_3min-fitcal.h5'

    dataset = mivit.helper.amisr.density(targtime, filename)
    pt = mivit.PlotMethod(cmap='jet', plot_type='scatter', label='PFISR Density', vmin=0., vmax=2.e11, s=10)
    data = mivit.DataVisualization(dataset, pt)

    plot = mivit.Visualize([data], map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-170,-135,50,75], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-150,'central_latitude':65})
    plot.one_map()


def main():
    test()
    # test_mango()
    # test_superdarn()
    # test_amisr()

if __name__ == '__main__':
    main()