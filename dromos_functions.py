import pandas as pd
import numpy as np
from math import pi, sin, cos, atan2
from bokeh.io import output_file, output_notebook, show
from bokeh.models import Line, GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, ColorBar, Range1d, PanTool, WheelZoomTool, BoxSelectTool
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5
import requests
from scipy.spatial.distance import cdist


def parse_thegeom(df):
	"""
	Function to parse out the longitude and latitude from the_geom variable from our dataframe.
	Arguments:
		- df: our dataframe
	Returns:
		- longitude, latitude: two vectors containing the coordinates from the_geom.
	"""
	if df.iloc[0].the_geom.split()[0] == 'POINT':
		lon = df.the_geom.apply(lambda x: x.split('(')[-1].split(')')[0].split()[0])
		lat = df.the_geom.apply(lambda x: x.split('(')[-1].split(')')[0].split()[1])

		return pd.to_numeric(lon), pd.to_numeric(lat)

	elif df.iloc[0].the_geom.split()[0] == 'MULTILINESTRING':
		lat_f = df.the_geom.apply(lambda x: x.split('(')[-1].split(')')[0].split(',')[0].split()[1])
		lon_f = df.the_geom.apply(lambda x: x.split('(')[-1].split(')')[0].split(',')[0].split()[0])
		lat_l = df.the_geom.apply(lambda x: x.split('(')[-1].split(')')[0].split(',')[-1].split()[1])
		lon_l = df.the_geom.apply(lambda x: x.split('(')[-1].split(')')[0].split(',')[-1].split()[0])

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

def closest_point(point, points):
    """ Find closest point from a list of points. 
    Arguments:
        point: the point for which we wish to find the closest point to.
        points: a list of points
    Returns: the closest point
    """
    return points[cdist([point], points).argmin()]

def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]

def coords_finder(address, api):
    """
    Function used to find the latitude and longitude coordinates of an address
    Arguments: 
        address: An address in NYC (string)
        api: the Google dev API key
    Returns:
        lat: latitude coordinate
        lon: longitude coordinate
    """
    add = address.replace(" ", "%20")
    url = 'https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input=' + add + '&inputtype=textquery&fields=formatted_address,name,geometry&key=' + api
    r = requests.get(url).json()
    lat = r['candidates'][0]['geometry']['location']['lat']
    lon = r['candidates'][0]['geometry']['location']['lng']
    return lat, lon

def ele_finder(card, api):
    """
    Function used to find the true elevation of a set of coordinates
    Arguments:
        card: a set of coords lat, lon
        api: Google Dev API key
    Returns:
        elevation: elevation of coordinates in meters.
    """
    lat = card[0]
    lon = card[1]
    url = 'https://maps.googleapis.com/maps/api/elevation/json?locations='+ str(lat) + ',' + str(lon) + '&key=' + api
    r = requests.get(url).json()
    elevation = r['results'][0]['elevation']
    return elevation


def map_plot(df, api):
    # plot the route
    map_options = GMapOptions(lat = 40.80, lng = -73.96, map_type = 'roadmap', zoom = 13)
    plot = GMapPlot( x_range= Range1d(), y_range= Range1d(), map_options=map_options)
    plot.api_key = api
    source = ColumnDataSource(data = dict(lat = df[0].values, lon = df[1].values))

    glyph = Line(x = 'lon', y = 'lat', line_color="#f46d43", line_width=6, line_alpha=0.6)
    circle = Circle(x = 'lon', y = 'lat', size = 5, fill_color = 'red', fill_alpha = 0.5, line_color = None)
    plot.add_glyph(source, circle)
    plot.add_glyph(source, glyph)

    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    return plot

def d_graph(start, finish, df):
	"""
	Dijkstra's Algorithm - using DFS to find the shortest path between two sets of coordinates, given a metric
	Arguments:
		- start: our starting coordinates
		- finish: our finish coordinates
		- df: our dataframe with all the information
	Returns:
		- search_df: a dataframe outlying the shortest route
	"""

    # figure out which way our route should be facing in order to narrow down our edges
    # latitude - wise
    if (start[0] - finish[0]) > 0:
        # means our route should be facing south hence we filter our dataframe for edges below start[0] + 300m and above finish[0] - 300m
        data = df[(df.first_x < start[0] + 0.003) & (df.last_x > finish[0] - 0.003)]
    else:
        # means our route should be facing north hence we filter our dataframe for edges above start[0] - 300m and below finish[0] + 300m
        data = df[(df.first_x > start[0] - 0.003) & (df.last_x < finish[0] + 0.003)]
    # longitude - wise
    if start[1] > finish[1]:
        # means our route should be facing east hence we filter our dataframe for edges left of start[1] - 250m and right of finish[1] + 250m
        data = data[(data.first_y < start[1] + 0.003) & (data.last_y > finish[1] - 0.003)]
    else:
        # means our route should be facing east hence we filter our dataframe for edges right of start[1] + 250m and left of finish[1] - 250m
        data = data[(data.first_y > start[1] - 0.003) & (data.last_y < finish[1] + 0.003)]

    # initialize our search dataframe
    initial_values = [(start, 0, np.nan)]
    search_df = pd.DataFrame(initial_values, columns = ['vertex', 'short_distance', 'prev_vertex'])

    # commence our graph build
    i = 0
    while True:

        # exit condition
        if i == search_df.shape[0]:
            break

        # assign the edge we check to v
        v = search_df.loc[i, 'vertex']

        # iterate every reachable vertex from edge v
        for next_v in data[data.coords_first == v].coords_last:

            # if it's not in our search dataframe, append it
            if not any(search_df.vertex == next_v):
                new_distance = data[(data.coords_first == v) & (data.coords_last == next_v)].distance.values[0] + search_df[search_df.vertex == v].short_distance.values[0]
                new_entry = {'vertex': next_v, 'short_distance': new_distance, 'prev_vertex': v}
                search_df = search_df.append(new_entry, ignore_index = True)

            # if it is in our dataframe, we only update it if its path is shorter than the one stored
            else:
                if search_df[search_df.vertex == next_v].short_distance.values[0] > (search_df[search_df.vertex == v].short_distance.values[0] + data[(data.coords_first == v) & (data.coords_last == next_v)].distance.values[0]):
                    new_distance = search_df[search_df.vertex == v].short_distance.values[0] + data[(data.coords_first == v) & (data.coords_last == next_v)].distance.values[0]
                    search_df.at[search_df[search_df.vertex == next_v].index[0], 'short_distance'] = new_distance
                    search_df.at[search_df[search_df.vertex == next_v].index[0], 'prev_vertex'] = v

        #increment i to repeat
        i += 1

    return search_df


def route_construction(graph_frame, beginning, finish):
	"""
	Function used to reformat the route dataframe constructed by our algorithm
	Arguments:
		- 
	"""

    i = 0
    route_list = []
    route_list.append(finish)
    v = finish
    while True:
        v = graph_frame[graph_frame.vertex == v].prev_vertex.values[0]
        route_list.append(v)

        if graph_frame[graph_frame.vertex == v].prev_vertex.values[0] == beginning:
            break

        i += 1

    return route_list
