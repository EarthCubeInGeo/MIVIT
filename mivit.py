# Mulit-Instrument VIsualization Toolkit (MIVIT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from apexpy import Apex
from apexpy.apex import ApexHeightError


class DataSet(object):
    def __init__(self,**kwargs):
        '''
            Valid keyword inputs:
            longitude
            latitude
            altitude
            site
            azimuth
            elevation
            ranges
            values
            instrument
            time_range
            cmap
            plot_type
            plot_kwargs
        '''
        # assign defaults
        self.plot_type = 'scatter'
        self.plot_kwargs = {}
        # assign input arguments
        self.__dict__.update(kwargs)

        # add the colormap to the plot_kwargs dict
        try:
            self.plot_kwargs['cmap'] = plt.get_cmap(self.cmap)
        except AttributeError:
            self.plot_kwargs['cmap'] = plt.get_cmap('jet')

        # parameters defining the ellipsoid earth (from WGS84)
        self.Req = 6378137.
        f = 1/298.257223563
        self.e2 = 2*f-f**2

        # find latitude/longitude arrays if site, azimuth, and elevation are given
        if not hasattr(self,'longitude'):
            if hasattr(self,'ranges'):
                self.latitude, self.longitude, self.altitude = self.azel2gd_ranges(self.site[0],self.site[1],self.site[2],self.azimuth,self.elevation,self.ranges)
            elif hasattr(self,'altitude'):
                self.latitude, self.longitude, self.altitude = self.azel2gd_alt(self.site[0],self.site[1],self.site[2],self.azimuth,self.elevation,self.altitude)



    def azel2gd_ranges(self,lat0,lon0,alt0,az,el,ranges):
        # convert azimuth and elevation to geodetic coordinates given ranges
        ranges = ranges*1000.
        el = el*np.pi/180.
        az = az*np.pi/180.
        x0, y0, z0 = self.geodetic_to_cartesian(lat0,lon0,alt0)

        ve = np.cos(el)*np.sin(az)
        vn = np.cos(el)*np.cos(az)
        vu = np.sin(el)

        vx, vy, vz = self.vector_geodetic_to_cartesian(vn,ve,vu,lat0,lon0,alt0)

        lat, lon, alt = self.cartesian_to_geodetic(x0+vx*ranges,y0+vy*ranges,z0+vz*ranges)

        return lat, lon, alt

    def azel2gd_alt(self,lat0,lon0,alt0,azimuth,elevation,proj_alt):
        # convert azimuth and elevation to geodetic coordinates given a projection altitude
        azimuth = azimuth*np.pi/180.
        elevation = elevation*np.pi/180.
        points = np.arange(0.,max(proj_alt)/np.sin(min(elevation)),1.)*1000.
        x0, y0, z0 = self.geodetic_to_cartesian(lat0,lon0,alt0)

        latitude = []
        longitude = []
        altitude = []

        for el, az, h in zip(elevation,azimuth,proj_alt):

            ve = np.cos(el)*np.sin(az)
            vn = np.cos(el)*np.cos(az)
            vu = np.sin(el)

            vx, vy, vz = self.vector_geodetic_to_cartesian(vn,ve,vu,lat0,lon0,alt0)

            lat, lon, alt = self.cartesian_to_geodetic(x0+vx*points,y0+vy*points,z0+vz*points)

            idx = (np.abs(alt-h)).argmin()

            latitude.append(lat[idx])
            longitude.append(lon[idx])
            altitude.append(alt[idx])

        latitude = np.array(latitude)
        longitude = np.array(longitude)
        altitude = np.array(altitude)

        return latitude, longitude, altitude


    def geodetic_to_cartesian(self,gdlat,gdlon,gdalt):
    #   Laundal, K. M. and Richmond, A. D. (2017). Magnetic Coordinate Systems. 
    #       Space Sci Rev, 206:27-59. doi: 10.1007/s11214-016-0275-y
        lam_gd = gdlat*np.pi/180.
        h = gdalt*1000.
        phi = gdlon*np.pi/180.
        rho = self.Req/np.sqrt(1-self.e2*np.sin(lam_gd)**2)
        x = (rho+h)*np.cos(lam_gd)*np.cos(phi)
        y = (rho+h)*np.cos(lam_gd)*np.sin(phi)
        z = (rho+h-self.e2*rho)*np.sin(lam_gd)
        return x, y, z

    def cartesian_to_geodetic(self,x,y,z):
    #   Heikkinen method taken from:
    #   Zhu, J. (1994). Conversion of Earth-centered Earth-fixed coordinates to geodetic coordinates. 
    #       IEEE Trans Aerosp Electron Syst, 30(3): 957-961. doi: 10.1109/7.303772

        a = self.Req
        b = a*np.sqrt(1-self.e2)
        r = np.sqrt(x**2+y**2)
        a2 = a**2
        b2 = b**2
        r2 = r**2
        z2 = z**2
        ep2 = (a2-b2)/b2
        F = 54*b2*z2
        G = r2+(1-self.e2)*z2-self.e2*(a2-b2)
        c = self.e2**2*F*r2/G**3
        s = np.cbrt(1+c+np.sqrt(c**2+2*c))
        P = F/(3*(s+1/s+1)**2*G**2)
        Q = np.sqrt(1+2*self.e2**2*P)
        r0 = -P*self.e2*r/(1+Q)+np.sqrt(a2/2*(1+1/Q)-(P*(1-self.e2)*z2)/(Q*(1+Q))-P*r2/2)
        U = np.sqrt((r-self.e2*r0)**2+z2)
        V = np.sqrt((r-self.e2*r0)**2+(1-self.e2)*z2)
        z0 = (b2*z)/(a*V)

        gdalt = U*(1-b2/(a*V))/1000.
        gdlat = np.arctan2(z+ep2*z0,r)*180./np.pi
        gdlon = 2*np.arctan2(r-x,y)*180./np.pi
        return gdlat, gdlon, gdalt

    def vector_geodetic_to_cartesian(self,vnd,ved,vud,gdlat,gdlon,gdalt):
    #   Laundal, K. M. and Richmond, A. D. (2017). Magnetic Coordinate Systems. 
    #       Space Sci Rev, 206:27-59. doi: 10.1007/s11214-016-0275-y

        lam_gd = gdlat*np.pi/180.
        h = gdalt*1000.
        phi = gdlon*np.pi/180.
        rho = self.Req/np.sqrt(1-self.e2*np.sin(lam_gd)**2)
        r = np.sqrt((rho+h)**2*np.cos(lam_gd)**2+(rho+h-self.e2*rho)**2*np.sin(lam_gd)**2)
        t = np.arccos((rho+h-self.e2*rho)*np.sin(lam_gd)/r)
        p = phi

        lam_gc = np.pi/2.-t
        b = lam_gd-lam_gc
        vr = vud*np.cos(-b)+vnd*np.sin(-b)
        vt = vud*np.sin(-b)-vnd*np.cos(-b)
        vp = ved

        vx = vr*np.sin(t)*np.cos(p)+vt*np.cos(t)*np.cos(p)-vp*np.sin(p)
        vy = vr*np.sin(t)*np.sin(p)+vt*np.cos(t)*np.sin(p)+vp*np.cos(p)
        vz = vr*np.cos(t)-vt*np.sin(t)

        return vx, vy, vz



class Visualize(object):
    def __init__(self,dataset_list,map_features=['gridlines','coastlines'],map_extent=None,map_proj='PlateCarree',map_proj_kwargs={}):
        self.dataset_list = dataset_list

        # define plot methods dictionary
        self.plot_methods = {'scatter':self.plot_scatter,'pcolormesh':self.plot_pcolormesh,'contour':self.plot_contour,'contourf':self.plot_contourf,'quiver':self.plot_quiver}
        self.no_colorbar = ['quiver','contour']

        self.map_features = map_features
        self.map_extent = map_extent

        self.map_proj = getattr(ccrs,map_proj)(**map_proj_kwargs)



    def one_map(self):
        # set up map
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        gs.update(left=0.01,right=0.99,top=0.99,bottom=0.1,hspace=0.1)
        ax = plt.subplot(gs[0],projection=self.map_proj)
        # set up background gridlines, coastlines, ect on map
        self.map_setup(ax)

        # define colorbar axes
        num_cbar = len([0 for dataset in self.dataset_list if dataset.plot_type not in self.no_colorbar])
        if num_cbar > 4:
            gs_cbar = gridspec.GridSpecFromSubplotSpec(int(np.ceil(num_cbar/2.)),2,subplot_spec=gs[1],hspace=50*gs.hspace,wspace=0.1)
        else:
            gs_cbar = gridspec.GridSpecFromSubplotSpec(num_cbar,1,subplot_spec=gs[1],hspace=50*gs.hspace,wspace=0.1)
        cbn = 0

        # plot image on map
        for dataset in self.dataset_list:
            f = self.plot_methods[dataset.plot_type](ax,dataset)
            if dataset.plot_type not in self.no_colorbar:
                cbar = fig.colorbar(f,cax=plt.subplot(gs_cbar[cbn]),orientation='horizontal')
                cbar.set_label('{} {}'.format(dataset.instrument,dataset.parameter))
                cbn+=1

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
    def plot_scatter(self,ax,dataset):
        if 's' not in dataset.plot_kwargs:
            dataset.plot_kwargs['s'] = 100*np.exp(-0.03*dataset.values.size)+0.1
        f = ax.scatter(dataset.longitude, dataset.latitude, c=dataset.values, transform=ccrs.Geodetic(), **dataset.plot_kwargs)
        return f

    def plot_pcolormesh(self,ax,dataset):
        f = ax.pcolormesh(dataset.longitude, dataset.latitude, dataset.values, transform=ccrs.PlateCarree(), **dataset.plot_kwargs)
        return f

    def plot_contour(self,ax,dataset):
        try:
            levels = dataset.plot_kwargs['levels']
            del dataset.plot_kwargs['levels']
        except KeyError:
            levels = 20
        f = ax.contour(dataset.longitude, dataset.latitude, dataset.values, levels, transform=ccrs.PlateCarree(), **dataset.plot_kwargs)
        ax.clabel(f,inline=1,fontsize=8,fmt='%1.1f')
        return f

    def plot_contourf(self,ax,dataset):
        try:
            levels = dataset.plot_kwargs['levels']
            del dataset.plot_kwargs['levels']
        except KeyError:
            levels = 20
        f = ax.contourf(dataset.longitude, dataset.latitude, dataset.values, levels, transform=ccrs.PlateCarree(), **dataset.plot_kwargs)
        return f

    def plot_quiver(self,ax,dataset):
        f = ax.quiver(dataset.longitude, dataset.latitude, dataset.values[0], dataset.values[1],transform=ccrs.PlateCarree(), **dataset.plot_kwargs)
        return f


def main():
    mivit()

if __name__ == '__main__':
    main()
