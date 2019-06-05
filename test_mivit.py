# test_mivit.py


import numpy as np
import datetime as dt
import configparser
from mivit import PlotMethod, DataSet, DataVisualization, Visualize
import helper
import copy



def test():

    targtime = dt.datetime(2017,5,28,6,35)

    # get SuperDARN data
    pt = PlotMethod(cmap='seismic',plot_type='pcolormesh',label='SuperDARN Velocity',vmin=-40,vmax=40)
    sd_data = []
    davitpy_kwargs = {'src':'local','fileType':'fitex','local_dirfmt':'./TestDataSets/SuperDARN/'}
    for rad in ['bks','fhe','fhw','kap','pgr','sas','wal']:
        sd = helper.SuperDARN_dataset(targtime,rad,davitpy_kwargs=davitpy_kwargs)
        sd_data.append(DataVisualization(sd, pt))


    # get mango data
    pt = PlotMethod(cmap='gist_gray',plot_type='pcolormesh', label='MANGO', vmin=0, vmax=255)
    mangopy_kwargs = {'datadir':'./TestDataSets/MANGO'}
    mango_data = helper.MANGO_dataset(targtime,mangopy_kwargs=mangopy_kwargs)
    mango = DataVisualization(mango_data, pt)



    # set up madrigalWeb credentials
    config = configparser.ConfigParser()
    config.read('config.ini')
    user_fullname = config.get('DEFAULT', 'MADRIGAL_FULLNAME')
    user_email = config.get('DEFAULT', 'MADRIGAL_EMAIL')
    user_affiliation = config.get('DEFAULT', 'MADRIGAL_AFFILIATION')
    user_info = {'fullname':user_fullname,'email':user_email,'affiliation':user_affiliation}



    # get GPS TEC
    pt = PlotMethod(cmap='magma',plot_type='contourf', label='GPS TEC', alpha=0.2, levels=25, vmin=0, vmax=20)
    pt2 = PlotMethod(cmap='magma',plot_type='contour', label='GPS TEC', levels=25, vmin=0, vmax=20)
    tec_dat = helper.GPSTEC_dataset(targtime,user_info)
    tec = DataVisualization(tec_dat,[pt,pt2])


    # # # get Millstone Hill data
    # # mlhtime = dt.datetime(2017,6,8,3,0,0)
    # # mlh = helper.MLHISR_dataset(mlhtime, user_info)

    # get Millston Hill FPI data
    fpi_g_dat = helper.MLHFPI_dataset(targtime, 'green', user_info)
    pt = PlotMethod(cmap='Greens', plot_type='scatter', label='FPI Tn', vmin=min(fpi_g_dat.values), vmax=max(fpi_g_dat.values), s=100)
    fpi_g = DataVisualization(fpi_g_dat, pt)

    pt = PlotMethod(color='green', plot_type='quiver', label='FPI Vn', width=0.002)
    fpi_vec_g_dat = helper.MLHFPIvec_dataset(targtime, 'green', user_info)
    fpi_vec_g = DataVisualization(fpi_vec_g_dat, pt)

    fpi_r_dat = helper.MLHFPI_dataset(targtime, 'red', user_info)
    pt = PlotMethod(cmap='Reds', plot_type='scatter', label='FPI Tn', vmin=min(fpi_r_dat.values), vmax=max(fpi_r_dat.values), s=30)    
    fpi_r = DataVisualization(fpi_r_dat, pt)

    pt = PlotMethod(color='red', plot_type='quiver', label='FPI Vn', width=0.002)
    fpi_vec_r_dat = helper.MLHFPIvec_dataset(targtime, 'red', user_info)
    fpi_vec_r = DataVisualization(fpi_vec_r_dat, pt)



    # get DMSP data
    pt = PlotMethod(cmap='jet',plot_type='scatter',label='DMSP Ni', vmin=0, vmax=3e10, s=20)
    dmsp_dat = helper.DMSP_dataset(targtime, user_info)
    dmsp = DataVisualization(dmsp_dat, pt)

    pt = PlotMethod(cmap='jet',plot_type='quiver',label='DMSP Vi', width=0.002)
    dmsp_dat = helper.DMSPvec_dataset(targtime, user_info)
    dmsp_vec = DataVisualization(dmsp_dat, pt)


    plot = Visualize([mango,tec,fpi_g,fpi_r,fpi_vec_g,fpi_vec_r,dmsp,dmsp_vec]+sd_data, map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    # plot = Visualize([dmsp,dmsp_vec], map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    plot.one_map()
    # plot.multi_map()





def main():
    test()

if __name__ == '__main__':
    main()