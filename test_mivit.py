# test_mivit.py


import numpy as np
import datetime as dt
import h5py
import cartopy.crs as ccrs
import configparser
from mivit import DataSet, Visualize
import helper
import copy



def test():


    # get SuperDARN data
    sdtime = dt.datetime(2016,10,1,17,0)
    sd_data = []
    for rad in ['sas','kap','pgr']:
        sd_data.append(helper.SuperDARN_dataset(sdtime,rad))


    # get mango data
    targtime = dt.datetime(2017,5,28,5,35)
    mango = helper.MANGO_dataset(targtime)




    # set up madrigalWeb credentials
    config = configparser.ConfigParser()
    config.read('config.ini')
    user_fullname = config.get('DEFAULT', 'MADRIGAL_FULLNAME')
    user_email = config.get('DEFAULT', 'MADRIGAL_EMAIL')
    user_affiliation = config.get('DEFAULT', 'MADRIGAL_AFFILIATION')
    user_info = {'fullname':user_fullname,'email':user_email,'affiliation':user_affiliation}



    # get GPS TEC
    tec1 = helper.GPSTEC_dataset(targtime,user_info)
    tec2 = copy.copy(tec1)
    tec2.plot_type='contour'
    tec2.plot_kwargs={'levels':tec1.plot_kwargs['levels']}



    # get Millstone Hill data
    mlhtime = dt.datetime(2017,6,8,3,0,0)
    mlh = helper.MLHISR_dataset(mlhtime, user_info)

    # get Millston Hill FPI data
    mlh_fpi = helper.MLHFPI_dataset(mlhtime, user_info)
    mlh_fpi_vec = helper.MLHFPIvec_dataset(mlhtime, user_info)



    plot = Visualize([mango,tec1,tec2,mlh,mlh_fpi,mlh_fpi_vec]+sd_data, map_features=['gridlines','coastlines','mag_gridlines'])
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