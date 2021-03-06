# Mulit-Instrument VIsualization Toolkit (MIVIT)
# My Inventory of Very Important Tadpoles

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from apexpy import Apex
from apexpy.apex import ApexHeightError


class Visualize(object):
    def __init__(self,datavis_list,map_features=['gridlines','coastlines'],map_extent=None,map_proj='PlateCarree',map_proj_kwargs={}):
        self.datavis_list = datavis_list
        self.check_time_range()

        # define plot methods dictionary
        self.plot_methods = {'scatter':self.plot_scatter,'pcolormesh':self.plot_pcolormesh,'contour':self.plot_contour,'contourf':self.plot_contourf,'quiver':self.plot_quiver}
        self.no_colorbar = ['quiver','contour']

        self.map_features = map_features
        self.map_extent = map_extent

        self.map_proj = getattr(ccrs,map_proj)(**map_proj_kwargs)


    def check_time_range(self, individual_datasets=False):
        starttimes = [dv.dataset.time_range[0] for dv in self.datavis_list]
        endtimes = [dv.dataset.time_range[1] for dv in self.datavis_list]
        starttime = min(starttimes)
        endtime = max(endtimes)
        print('Datasets range from {:%Y-%m-%d %H:%M:%S} to {:%Y-%m-%d %H:%M:%S}, covering {} minutes.'.format(starttime, endtime, (endtime-starttime).total_seconds()/60.))
        if individual_datasets:
            for datavis in self.datavis_list:
                print('{}: {:%Y-%m-%d %H:%M:%S} - {:%Y-%m-%d %H:%M:%S}'.format(datavis.dataset.name, datavis.dataset.time_range[0], datavis.dataset.time_range[1]))

    def one_map(self):
        # set up map
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        gs.update(left=0.01,right=0.99,top=0.99,bottom=0.1,hspace=0.1)
        ax = plt.subplot(gs[0],projection=self.map_proj)
        # set up background gridlines, coastlines, ect on map
        self.map_setup(ax)

        # plot image on map
        for datavis in self.datavis_list:
            for pm in datavis.plotmethods:
                self.plot_methods[pm.plot_type](ax,datavis.dataset,pm)

        # get list of unique colorbars
        # colorbars = [[pm for pm in dataset.plotmethods if pm.plot_type not in self.no_colorbar] for dataset in self.dataset_list]
        colorbars = []
        for datavis in self.datavis_list:
            for pm in datavis.plotmethods:
                if pm.plot_type not in self.no_colorbar:
                    colorbars.append(pm)
        colorbars = set(colorbars)

        # define colorbar axes
        num_cbar = len(colorbars)
        if num_cbar > 4:
            gs_cbar = gridspec.GridSpecFromSubplotSpec(int(np.ceil(num_cbar/2.)),2,subplot_spec=gs[1],hspace=50*gs.hspace,wspace=0.1)
        else:
            gs_cbar = gridspec.GridSpecFromSubplotSpec(num_cbar,1,subplot_spec=gs[1],hspace=50*gs.hspace,wspace=0.1)

        # create each colorbar on plot
        for i, cb in enumerate(colorbars):
            sm = plt.cm.ScalarMappable(cmap=cb.plot_kwargs['cmap'],norm=plt.Normalize(vmin=cb.plot_kwargs['vmin'],vmax=cb.plot_kwargs['vmax']))
            sm._A = []
            cbar = plt.colorbar(sm, orientation='horizontal', cax=plt.subplot(gs_cbar[i]))
            cbar.set_label(cb.label)

        plt.savefig('mivit_test.png')
        plt.show()

    def multi_map(self):
        # set up figure
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(int(np.ceil(len(self.dataset_list)/2.)),2)
        gs.update(left=0.01,right=0.9,wspace=0.2,hspace=0.01)

        for n, dataset in enumerate(self.dataset_list):
            ax = plt.subplot(gs[n],projection=self.map_proj)
            # set up background gridlines, coastlines, ect on map
            self.map_setup(ax)

            f = self.plot_methods[dataset.plot_type](ax,dataset)
            ax.set_title('{} {}'.format(dataset.instrument,dataset.parameter))
            if dataset.plot_type not in self.no_colorbar:
                pos = ax.get_position()
                cax = fig.add_axes([pos.x1+0.01,pos.y0,0.015,(pos.y1-pos.y0)])
                fig.colorbar(f,cax=cax)

        plt.show()

    def map_setup(self, ax):

        if self.map_extent:
            ax.set_extent(self.map_extent,crs=ccrs.Geodetic())
        xticks = []
        yticks = []
        xticklabels = []
        yticklabels = []
        xtickcolor = []
        ytickcolor = []

        if 'gridlines' in self.map_features:
            # latlines = range(20,80,10)
            # lonlines = range(200,330,30)
            latlines = range(-90,90,10)
            lonlines = range(0,360,30)
            ax.gridlines(xlocs=lonlines,ylocs=latlines,crs=ccrs.Geodetic())

            gridlines = [np.array([np.linspace(0,360,100),np.full(100,lat)]) for lat in latlines] + [np.array([np.full(100,lon),np.linspace(-90,90,100)]) for lon in lonlines]
            gridlabels = ['{} N'.format(lat) for lat in latlines] + ['{} E'.format(lon) for lon in lonlines]
            xt, yt, xtl, ytl = self.map_ticks(ax,gridlines,gridlabels)
            xticks.extend(xt)
            yticks.extend(yt)
            xticklabels.extend(xtl)
            yticklabels.extend(ytl)
            xtickcolor.extend(['black']*len(xt))
            ytickcolor.extend(['black']*len(yt))

        if 'mag_gridlines' in self.map_features:
            # mlatlines = range(20,80,10)
            # mlonlines = range(300,390,30)
            mlatlines = range(-90,90,10)
            mlonlines = range(0,360,30)
            gridlines, gridlabels = self.magnetic_gridlines(ax,xlocs=mlonlines,ylocs=mlatlines)
            xt, yt, xtl, ytl = self.map_ticks(ax,gridlines,gridlabels)
            xticks.extend(xt)
            yticks.extend(yt)
            xticklabels.extend(xtl)
            yticklabels.extend(ytl)
            xtickcolor.extend(['red']*len(xt))
            ytickcolor.extend(['red']*len(yt))

        # put gridline markers on plot
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        for t, c in zip(ax.xaxis.get_ticklabels(),xtickcolor):
            t.set_color(c)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        for t, c in zip(ax.yaxis.get_ticklabels(),ytickcolor):
            t.set_color(c)


        if 'coastlines' in self.map_features:
            ax.coastlines()

        if 'statelines' in self.map_features:
            ax.add_feature(cfeature.STATES)



    def magnetic_gridlines(self,ax,xlocs=None,ylocs=None):
        if not xlocs:
            xlocs = np.arange(0,360,30)
        if not ylocs:
            ylocs = np.arange(-90,90,15)
        A = Apex(2017)
        gridlines = []
        gridlabels = []
        for mlat in ylocs:
            try:
                gdlat, gdlon = A.convert(mlat,np.linspace(0,360,100),'apex','geo',height=0)
                ax.plot(gdlon,gdlat,transform=ccrs.Geodetic(),color='pink',linewidth=1.)
                gridlines.append(np.array([gdlon,gdlat]))
                gridlabels.append('{} N'.format(mlat))
            except ApexHeightError as e:
                continue
        for mlon in xlocs:
            try:
                gdlat, gdlon = A.convert(np.linspace(-90,90,100),mlon,'apex','geo',height=0)
                ax.plot(gdlon,gdlat,transform=ccrs.Geodetic(),color='pink',linewidth=1.)
                gridlines.append(np.array([gdlon,gdlat]))
                gridlabels.append('{} E'.format(mlon))
            except ApexHeightError as e:
                continue

        return gridlines, gridlabels



    def map_ticks(self, ax, lines, labels):

        xticks = []
        yticks = []
        xticklabels = []
        yticklabels = []

        for line, label in zip(lines,labels):
            tick_locations = self.tick_location(ax,line)

            if label.endswith('E') or label.endswith('W'):
                xticks.extend(tick_locations['bottom'])
                xticklabels.extend([label]*len(tick_locations['bottom']))

            if label.endswith('N') or label.endswith('S'):
                yticks.extend(tick_locations['left'])
                yticklabels.extend([label]*len(tick_locations['left']))

        return xticks, yticks, xticklabels, yticklabels


    def tick_location(self,ax,line,edges_with_ticks=['left','bottom']):

        # # need do do something to solve the crossing dateline issue
        # s = np.argwhere(line[0,1:]-line[0,:-1]<-180.)
        # if s:
        #     s = int(s.flatten()[0])+1
        #     line = np.concatenate((line[:,s:],line[:,:s]),axis=1)

        # convert line from geodetic coordinates to map projection coordinates
        line_map = ax.projection.transform_points(ccrs.Geodetic(),line[0],line[1]).T

        # parameters specific for finding ticks on each edge of plot
        edge_params = {'left':{'axis_i':0,'axis_d':1,'edge':ax.viewLim.x0,'bounds':[ax.viewLim.y0,ax.viewLim.y1]},
                       'right':{'axis_i':0,'axis_d':1,'edge':ax.viewLim.x1,'bounds':[ax.viewLim.y0,ax.viewLim.y1]},
                       'bottom':{'axis_i':1,'axis_d':0,'edge':ax.viewLim.y0,'bounds':[ax.viewLim.x0,ax.viewLim.x1]},
                       'top':{'axis_i':1,'axis_d':0,'edge':ax.viewLim.y1,'bounds':[ax.viewLim.x0,ax.viewLim.x1]}}

        # initialize empty dictionary to be returned wiht tick locations
        tick_locations = {}

        for e in edges_with_ticks:
            axis = edge_params[e]['axis_i']
            axisd = edge_params[e]['axis_d']
            edge = edge_params[e]['edge']
            bounds = edge_params[e]['bounds']
            tick_locations[e] = []

            # find indicies where the line crosses the edge
            line_map_shift = line_map[axis]-edge
            args = np.argwhere(line_map_shift[:-1]*line_map_shift[1:]<0).flatten()

            # for each crossing, interpolate to edge to find the tick location
            for a in args:
                l = line_map[0:2,a:a+2]
                l = l[:,l[axis].argsort()]

                tick = np.interp(edge,l[axis],l[axisd])
                # if tick is located within the plot, add it to list of tick locations
                if tick>bounds[0] and tick<bounds[1]:
                    tick_locations[e].append(tick)

        return tick_locations



    # functions for plotting based on different methods
    def plot_scatter(self,ax,dataset,plotmethod):
        if 's' not in plotmethod.plot_kwargs:
            plotmethod.plot_kwargs['s'] = 100*np.exp(-0.03*dataset.values.size)+0.1

        f = ax.scatter(dataset.longitude, dataset.latitude, c=dataset.values, transform=ccrs.Geodetic(), **plotmethod.plot_kwargs)
        return f

    def plot_pcolormesh(self,ax,dataset,plotmethod):
        f = ax.pcolormesh(dataset.longitude, dataset.latitude, dataset.values, transform=ccrs.PlateCarree(), **plotmethod.plot_kwargs)
        return f

    def plot_contour(self,ax,dataset,plotmethod):
        f = ax.contour(dataset.longitude, dataset.latitude, dataset.values, plotmethod.levels, transform=ccrs.PlateCarree(), **plotmethod.plot_kwargs)
        ax.clabel(f,inline=1,fontsize=8,fmt='%1.1f')
        return f

    def plot_contourf(self,ax,dataset,plotmethod):
        f = ax.contourf(dataset.longitude, dataset.latitude, dataset.values, plotmethod.levels, transform=ccrs.PlateCarree(), **plotmethod.plot_kwargs)
        return f

    def plot_quiver(self,ax,dataset,plotmethod):
        f = ax.quiver(dataset.longitude, dataset.latitude, dataset.values[0], dataset.values[1],transform=ccrs.PlateCarree(), **plotmethod.plot_kwargs)
        return f


# def main():
#     mivit()

# if __name__ == '__main__':
#     main()
