# dataset.py

import numpy as np
import pymap3d as pm

class DataSet(object):
    def __init__(self,**kwargs):
        '''
        A single data set object.  Datasets must have:
            - spatial coordinates
            - data values at each coordinate
            - the range of time over which the dataset values are valid

        Parameters:
            - values: Array of data values.  Can be either scalar values or 3 components of vectors.
            - time_range: [starttime, endtime] specifying the range of times over which the data set is valid.
            - latitude (optional): Array of latitude coordinates for each data point
            - longitude (optional): Array of longitude coordinates for each data point
            - altitude (optional): Array of altitude coordinates for each data point
            - site (optional): [lat, lon, alt] of the instrument site
            - azimuth (optional): Array of azimuth values for each data point
            - elevation (optional): Array of elevation values for each data point
            - slantrange (optional): Array of slant range values for each data point

        A DataSet object can be initialized a variety of different ways, based on the input keyword arguments povided
        Can be initialized with EITHER:
        latitude, longitude, altitude
        OR
        site, azimuth, elevation, ranges
        OR
        site, azimuth, elevation, altitude

        Vector components:
        Geodetic E, N, Z
        Satelite forward, left, vertical
        If satellite components are input, set sat_comp=True
        '''

        # convert input coordinates to latitude, longitude, altitude arrays
        if all(k in kwargs for k in ('latitude', 'longitude', 'altitude')):
            pass
        elif all(k in kwargs for k in ('site','azimuth','elevation','slantrange')):
            # kwargs['latitude'], kwargs['longitude'], kwargs['altitude'] = self.azel2gd_ranges(kwargs['site'][0],kwargs['site'][1],kwargs['site'][2],kwargs['azimuth'],kwargs['elevation'],kwargs['ranges'])
            kwargs['latitude'], kwargs['longitude'], kwargs['altitude'] = pm.aer2geodetic(kwargs['azimuth'], kwargs['elevation'], kwargs['range'], kwargs['site'][0],kwargs['site'][1],kwargs['site'][2])
        elif all(k in kwargs for k in ('site','azimuth','elevation','altitude')):
            # kwargs['latitude'], kwargs['longitude'], kwargs['altitude'] = self.azel2gd_alt(kwargs['site'][0],kwargs['site'][1],kwargs['site'][2],kwargs['azimuth'],kwargs['elevation'],kwargs['altitude'])
            kwargs['latitude'], kwargs['longitude'], kwargs['altitude'] = self.projalt(kwargs['site'], kwargs['azimuth'], kwargs['elevation'], kwargs['altitude'])
        else:
            raise ValueError('Incorrect set of input arguments!')

        # if input vector in satelite forward, left, vertical components, convert to geodetic east, north, up
        if 'sat_comp' in kwargs:
            kwargs['values'] = self.convert_sat_comp_to_geodetic(kwargs['values'], kwargs['latitude'], kwargs['longitude'], kwargs['altitude'])

        # assign input arguments
        self.__dict__.update(kwargs)



    def convert_sat_comp_to_geodetic(self, vector, latitude, longitude, altitude):
        '''
        Convert vectors with satellite forward, left, up components to geodetic East, North, Up
        '''

        x, y, z = pm.geodetic2ecef(latitude, longitude, altitude)

        sat_position = np.array([x, y, z]).T
        sat_pos_dif = np.concatenate((np.array([np.nan,np.nan,np.nan])[None,:],sat_position[2:,:]-sat_position[:-2,:],np.array([np.nan,np.nan,np.nan])[None,:]),axis=0)
        forwECEF = sat_pos_dif/np.linalg.norm(sat_pos_dif,axis=-1)[:,None]

        ve, vn, vz = pm.uvw2enu(forwECEF[:,0], forwECEF[:,1], forwECEF[:,2], latitude, longitude)

        forw = np.array([ve, vn, vz]).T
        vert = np.tile(np.array([0,0,1]),(len(forw),1))
        left = np.cross(vert,forw)

        R = np.array([forw.T, left.T, vert.T])
        Vgd = np.einsum('ijk,...ik->...ij',R.T,vector.T).T

        return Vgd




    def projalt(self,site,azimuth,elevation,proj_alt):
        # convert azimuth and elevation to geodetic coordinates given a projection altitude

        points = np.arange(0.,max(proj_alt)/np.sin(min(elevation)),1.)*1000.

        lat, lon, alt = pm.aer2geodetic(azimuth, elevation, points[:,None], site[0], site[1], site[2])

        # find index closeset to the projection altitude
        idx = np.argmin(np.abs(alt-proj_alt*1000.),axis=0)

        # select altitude at correct index for each beam (fancy indexing with tuples to avoid for loops)
        latitude = lat[tuple(idx),tuple(np.arange(len(proj_alt)))]
        longitude = lon[tuple(idx),tuple(np.arange(len(proj_alt)))]
        altitude = alt[tuple(idx),tuple(np.arange(len(proj_alt)))]

        return latitude, longitude, altitude
