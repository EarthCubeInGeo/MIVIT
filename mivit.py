# Mulit-Instrument VIsualization Toolkit (MIVIT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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
        '''
        # assign defaults
        self.plot_type = 'scatter'
        # assign input arguments
        self.__dict__.update(kwargs)

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
    def __init__(self,dataset_list):
        self.dataset_list = dataset_list

        # define plot methods dictionary
        self.plot_methods = {'scatter':self.plot_scatter,'pcolormesh':self.plot_pcolormesh,'contour':self.plot_contour,'contourf':self.plot_contourf,'quiver':self.plot_quiver}
        self.no_colorbar = ['quiver','contour']



    def one_map(self):
        # set up map
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        gs.update(left=0.01,right=0.99,top=0.99,bottom=0.1,hspace=0.01)
        map_proj = ccrs.LambertConformal(central_longitude=-110.,central_latitude=40.0)
        ax = plt.subplot(gs[0],projection=map_proj)
        ax.coastlines()
        ax.gridlines()
        ax.add_feature(cfeature.STATES)
        ax.set_extent([225,300,25,50])

        # define colorbar axes
        num_cbar = len([0 for dataset in self.dataset_list if dataset.plot_type not in self.no_colorbar])
        gs_cbar = gridspec.GridSpecFromSubplotSpec(num_cbar,1,subplot_spec=gs[1],hspace=1.)
        cbn = 0

        # plot image on map
        for dataset in self.dataset_list:
            f = self.plot_methods[dataset.plot_type](ax,dataset)
            if dataset.plot_type not in self.no_colorbar:
                cbar = fig.colorbar(f,cax=plt.subplot(gs_cbar[cbn]),orientation='horizontal')
                cbar.set_label('{} {}'.format(dataset.instrument,dataset.parameter))
                cbn+=1

        plt.show()

    def multi_map(self):
        # set up figure
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(int(np.ceil(len(self.dataset_list)/2.)),2)
        gs.update(left=0.01,right=0.9,wspace=0.2,hspace=0.01)
        map_proj = ccrs.LambertConformal(central_longitude=-110.,central_latitude=40.)

        for n, dataset in enumerate(self.dataset_list):
            ax = plt.subplot(gs[n],projection=map_proj)
            ax.coastlines()
            ax.gridlines()
            ax.add_feature(cfeature.STATES)
            ax.set_extent([225,300,25,50])

            f = self.plot_methods[dataset.plot_type](ax,dataset)
            ax.set_title('{} {}'.format(dataset.instrument,dataset.parameter))
            if dataset.plot_type not in self.no_colorbar:
                pos = ax.get_position()
                cax = fig.add_axes([pos.x1+0.01,pos.y0,0.015,(pos.y1-pos.y0)])
                fig.colorbar(f,cax=cax)

        plt.show()


    # functions for plotting based on different methods
    def plot_scatter(self,ax,dataset):
        point_size = 100*np.exp(-0.03*dataset.values.size)+0.1
        f = ax.scatter(dataset.longitude, dataset.latitude, c=dataset.values, s=point_size, cmap=plt.get_cmap(dataset.cmap), transform=ccrs.Geodetic())
        return f

    def plot_pcolormesh(self,ax,dataset):
        f = ax.pcolormesh(dataset.longitude, dataset.latitude, dataset.values, cmap=plt.get_cmap(dataset.cmap), transform=ccrs.PlateCarree())
        return f

    def plot_contour(self,ax,dataset):
        levels = 20
        f = ax.contour(dataset.longitude, dataset.latitude, dataset.values, levels, cmap=plt.get_cmap(dataset.cmap), transform=ccrs.PlateCarree())
        ax.clabel(f,inline=1,fontsize=8,fmt='%1.1f')
        return f

    def plot_contourf(self,ax,dataset):
        f = ax.contourf(dataset.longitude, dataset.latitude, dataset.values, cmap=plt.get_cmap(dataset.cmap), transform=ccrs.PlateCarree())
        return f

    def plot_quiver(self,ax,dataset):
        f = ax.quiver(dataset.longitude, dataset.latitude, dataset.values[0], dataset.values[1],transform=ccrs.PlateCarree())
        return f


def main():
    mivit()

if __name__ == '__main__':
    main()
