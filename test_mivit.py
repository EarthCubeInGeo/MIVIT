# test_mivit.py


import numpy as np
import datetime as dt
import configparser
from mivit import PlotMethod, DataSet, DataVisualization, Visualize
import helper
import copy



def test():


    # get SuperDARN data
    pt = PlotMethod(cmap='seismic',plot_type='pcolormesh',label='SuperDARN Velocity',vmin=-40,vmax=40)
    sdtime = dt.datetime(2017,5,28,5,35)
    sd_data = []
    davitpy_kwargs = {'src':'local','fileType':'fitex','local_dirfmt':'./TestDataSets/SuperDARN/'}
    for rad in ['bks','fhe','fhw','kap','pgr','sas','wal']:
        sd = helper.SuperDARN_dataset(sdtime,rad,davitpy_kwargs=davitpy_kwargs)
        sd_data.append(DataVisualization(sd, pt))


    # get mango data
    pt = PlotMethod(cmap='gist_gray',plot_type='pcolormesh', label='MANGO', vmin=0, vmax=255)
    targtime = dt.datetime(2017,5,28,5,35)
    mangopy_kwargs = {'datadir':'./TestDataSets/MANGO'}
    mango_data = helper.MANGO_dataset(targtime,mangopy_kwargs=mangopy_kwargs)
    mango = DataVisualization(mango_data, pt)
    # mango.plot_kwargs['vmax'] = 200




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
    # tec1.plot_kwargs = {'alpha':0.2, 'levels':25,'vmin':0,'vmax':20}
    # tec2 = copy.copy(tec1)
    # tec2.plot_type='contour'
    # tec2.plot_kwargs={'levels':tec1.plot_kwargs['levels'],'vmin':tec1.plot_kwargs['vmin'],'vmax':tec1.plot_kwargs['vmax']}



    # # # get Millstone Hill data
    # # mlhtime = dt.datetime(2017,6,8,3,0,0)
    # # mlh = helper.MLHISR_dataset(mlhtime, user_info)

    # get Millston Hill FPI data
    pt = PlotMethod(cmap='Greens', plot_type='scatter', label='FPI Tn', vmin=280, vmax=310, s=100)
    fpi_g_dat = helper.MLHFPI_dataset(targtime, 'green', user_info)
    fpi_g = DataVisualization(fpi_g_dat, pt)

    # mlh_fpi_g.plot_kwargs['cmap'] = 'Greens'
    # mlh_fpi_g.plot_kwargs['s'] = 40

    pt = PlotMethod(color='green', plot_type='quiver', label='FPI Vn', width=0.002)
    fpi_vec_g_dat = helper.MLHFPIvec_dataset(targtime, 'green', user_info)
    fpi_vec_g = DataVisualization(fpi_vec_g_dat, pt)
    # mlh_fpi_vec_g.plot_kwargs['color']='green'

    pt = PlotMethod(cmap='Reds', plot_type='scatter', label='FPI Tn', vmin=1100, vmax=1300, s=30)    
    fpi_r_dat = helper.MLHFPI_dataset(targtime, 'red', user_info)
    fpi_r = DataVisualization(fpi_r_dat, pt)
    # mlh_fpi_r.plot_kwargs['cmap'] = 'Reds'

    pt = PlotMethod(color='red', plot_type='quiver', label='FPI Vn', width=0.002)
    fpi_vec_r_dat = helper.MLHFPIvec_dataset(targtime, 'red', user_info)
    fpi_vec_r = DataVisualization(fpi_vec_r_dat, pt)
    # mlh_fpi_vec_r.plot_kwargs['color']='red'

    # # # get DMSP data
    # # dmsp = helper.DMSP_dataset(targtime, user_info)



    plot = Visualize([mango,tec,fpi_g,fpi_r,fpi_vec_g,fpi_vec_r]+sd_data, map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    # plot = Visualize([mango]+sd_data, map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    plot.one_map()
    # plot.multi_map()





def main():
    test()

if __name__ == '__main__':
    main()