# test_mivit.py


import numpy as np
import datetime as dt
import configparser
from mivit import PlotType, DataSet, Visualize
import helper
import copy



def test():


    # get SuperDARN data
    pt = PlotType(cmap='seismic',plot_type='pcolormesh',label='SuperDARN Velocity',plot_kwargs={'vmin':-40,'vmax':40})
    sdtime = dt.datetime(2017,5,28,5,35)
    sd_data = []
    davitpy_kwargs = {'src':'local','fileType':'fitex','local_dirfmt':'./TestDataSets/SuperDARN/'}
    for rad in ['bks','fhe','fhw','kap','pgr','sas','wal']:
        sd_data.append(helper.SuperDARN_dataset(sdtime,rad,davitpy_kwargs=davitpy_kwargs,plot_type=pt))


    # get mango data
    pt = PlotType(cmap='gist_gray',plot_type='pcolormesh', label='MANGO')
    targtime = dt.datetime(2017,5,28,5,35)
    mangopy_kwargs = {'datadir':'./TestDataSets/MANGO'}
    mango = helper.MANGO_dataset(targtime,mangopy_kwargs=mangopy_kwargs,plot_type=pt)
    # mango.plot_kwargs['vmax'] = 200




    # # set up madrigalWeb credentials
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    # user_fullname = config.get('DEFAULT', 'MADRIGAL_FULLNAME')
    # user_email = config.get('DEFAULT', 'MADRIGAL_EMAIL')
    # user_affiliation = config.get('DEFAULT', 'MADRIGAL_AFFILIATION')
    # user_info = {'fullname':user_fullname,'email':user_email,'affiliation':user_affiliation}



    # # get GPS TEC
    # tec1 = helper.GPSTEC_dataset(targtime,user_info)
    # tec1.plot_kwargs = {'alpha':0.2, 'levels':25,'vmin':0,'vmax':20}
    # tec2 = copy.copy(tec1)
    # tec2.plot_type='contour'
    # tec2.plot_kwargs={'levels':tec1.plot_kwargs['levels'],'vmin':tec1.plot_kwargs['vmin'],'vmax':tec1.plot_kwargs['vmax']}



    # # # get Millstone Hill data
    # # mlhtime = dt.datetime(2017,6,8,3,0,0)
    # # mlh = helper.MLHISR_dataset(mlhtime, user_info)

    # # get Millston Hill FPI data
    # mlh_fpi_g = helper.MLHFPI_dataset(targtime, 'green', user_info)
    # mlh_fpi_g.plot_kwargs['cmap'] = 'Greens'
    # mlh_fpi_g.plot_kwargs['s'] = 40
    # mlh_fpi_vec_g = helper.MLHFPIvec_dataset(targtime, 'green', user_info)
    # mlh_fpi_vec_g.plot_kwargs['color']='green'
    # mlh_fpi_r = helper.MLHFPI_dataset(targtime, 'red', user_info)
    # mlh_fpi_r.plot_kwargs['cmap'] = 'Reds'
    # mlh_fpi_vec_r = helper.MLHFPIvec_dataset(targtime, 'red', user_info)
    # mlh_fpi_vec_r.plot_kwargs['color']='red'

    # # # get DMSP data
    # # dmsp = helper.DMSP_dataset(targtime, user_info)



    # plot = Visualize([mango,tec1,tec2,mlh_fpi_r,mlh_fpi_g,mlh_fpi_vec_r,mlh_fpi_vec_g]+sd_data, map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    plot = Visualize([mango]+sd_data, map_features=['gridlines','coastlines','mag_gridlines'], map_extent=[-130,-65,20,50], map_proj='LambertConformal', map_proj_kwargs={'central_longitude':-100,'central_latitude':35})
    plot.one_map()
    # plot.multi_map()





def main():
    test()

if __name__ == '__main__':
    main()