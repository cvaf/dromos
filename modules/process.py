import pandas as pd
import numpy as np
from math import pi, sin, cos, atan2
from scipy.spatial.distance import cdist


def point_parser(x, i):
    """
    Parse out the latitude or longitude from a "POINT" the_geom
    Arguments:
        - x: the_geom
        - i: indicates whether we want the the longitude or latitude
    Returns:
        - the corresponding longitude,latitude
    """
    return x.split('(')[-1].split(')')[0].split()[i]

def multiline_parser(x, first, second):
    """
    Parse out the latitude or longitude from a "MULTILINESTRING" the_geom
    Arguments:
        - x: the_geom
        - first, second: indices to parse out the first or second set of coordinates, latitude or longitude
    Returns:
        - corresponding lat,lon for first or second set of coordinates
    """
    return x.split('(')[-1].split(')')[0].split(',')[first].split()[second]

def coords_parser(coords):
    """
    Parse out the longitude and latitude from the_geom variable from our dataframe.
    Arguments:
        - coords: array of coordinates to parse out
    Returns:
        - longitude, latitude: two vectors containing the coordinates from the_geom.
    """
    coordinate_type = coords[0].split()[0]

    if coordinate_type == 'POINT':
        lon = coords.apply(point_parser, args=(0,))
        lat = coords.apply(point_parser, args=(1,))
        return pd.to_numeric(lon), pd.to_numeric(lat)

    elif coordinate_type == 'MULTILINESTRING':
        lat_f = coords.apply(multiline_parser, args=(0,1,))
        lon_f = coords.apply(multiline_parser, args=(0,0,))
        lat_l = coords.apply(multiline_parser, args=(-1,1,))
        lon_l = coords.apply(multiline_parser, args=(-1,0,))

        return pd.to_numeric(lat_f), pd.to_numeric(lon_f), pd.to_numeric(lat_l), pd.to_numeric(lon_l)


def meters_dist(row):
    """
    Takes a row as input and outputs the distance between the two coordinates in meters.
    Args:
        row: a row in our dataframe that consists two sets of coordinates.
    Returns: the distance in meters between the two sets of coordinates.
    """
    lat1 = row['coords_first'][0]
    lon1 = row['coords_first'][1]
    lat2 = row['coords_last'][0]
    lon2 = row['coords_last'][1]
    R = 6378.137
    dLat = lat2 * pi / 180 - lat1 * pi / 180
    dLon = lon2 * pi / 180 - lon1 * pi / 180
    a = sin(dLat/2) * sin(dLat/2) + cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000


def closest_elevation(point, df_ele):
    """
    Given a point, this function finds the closest elevation from our elevation dataframe.
    Arguments:
        - point: a set of coordinates for which we are missing the elevation for
        - df_ele: our elevation dataframe
    Returns:
        - an approximation of the elevation for that set of coordinates
    """
    points = list(df_ele['coords_first'])
    ele = df_ele.loc[cdist([point], points).argmin(), 'elevation']
    return ele

def fill_elevations(coords, df_ele):
    """
    Function used to find the closest elevation for a list of observations with no elevation
    Arguments:
        - coords: our coordinates with missing elevations
        - df_ele: our elevation dataframe
    Returns: 
        - elevations: a list of elevations for those data points
    """
    elevations = [closest_elevation(i, df_ele) for i in coords]
    return elevations





