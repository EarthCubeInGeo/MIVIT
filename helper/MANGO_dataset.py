# mango_dataset.py

from mangopy.mosaic import Mosaic
from mivit import DataSet

def MANGO_dataset(targtime):
    m = Mosaic()
    mosaic, mosaic_lat, mosaic_lon = m.create_mosaic(targtime)
    dataset = DataSet(longitude=mosaic_lon,latitude=mosaic_lat,values=mosaic,cmap='gist_gray',plot_type='pcolormesh', instrument='MANGO', parameter='Brightness')
    return dataset
