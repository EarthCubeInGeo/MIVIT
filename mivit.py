# Mulit-Instrument VIsualization Toolkit (MIVIT)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class DataSet(object):
    def __init__(self,longitude=None,latitude=None,altitude=None,values=None,instrument=None,time_range=None,cmap=None):
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.values = values
        self.instrument = instrument
        self.time_range = time_range
        self.cmap = cmap

class Visualize(object):
    def __init__(self,dataset_list):
        self.dataset_list = dataset_list



    def one_map(self):
        # set up map
        fig = plt.figure(figsize=(15,10))
        map_proj = ccrs.LambertConformal(central_longitude=-110.,central_latitude=40.0)
        ax = fig.add_subplot(111,projection=map_proj)
        ax.coastlines()
        ax.gridlines()
        ax.add_feature(cfeature.STATES)
        # ax.set_extent([225,300,25,50])

        # # plot image on map
        for dataset in self.dataset_list:
            ax.scatter(dataset.longitude, dataset.latitude, c=dataset.values, s=0.1, cmap=plt.get_cmap(dataset.cmap), transform=ccrs.Geodetic())

        plt.show()




def main():
    mivit()

if __name__ == '__main__':
    main()
