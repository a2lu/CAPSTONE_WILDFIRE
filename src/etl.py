import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
from functools import reduce
from operator import iconcat, itemgetter
# from funcs import convertDate


def boundsBuffer(x, buffer=0.075):
    """
    Returns a Polygon geometry that represents a bounding box with a default 7.5% buffer
    """
    minx, miny, maxx, maxy = x
    buffer_x, buffer_y = np.abs(buffer*(maxx-minx)), np.abs(buffer*(maxy-miny))
    minx -= buffer_x
    maxx += buffer_x
    miny -= buffer_y
    maxy += buffer_y

    coords = ((minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny))
    return Polygon(coords)


def convertDate(date):
    """
    Converts EE.Date or unix date to Y-M-D formst
    """
    if isinstance(date, ee.Date):
        date = date.getInfo()["value"]

    return datetime.utcfromtimestamp(date/1000).strftime("%Y-%m-%d")# %H:%M:%S')


def sizeCode(x):
    if x < 5000:
        return "A"
    elif 5000 <= x < 10000:
        return "G"
    elif 10000 <= x < 50000:
        return "H"
    elif 50000 <= x < 100000:
        return "I"
    elif 100000 <= x:
        return "J+"



def genSamplePoints(feature, gridScale, pointScale, seed):
    """
    Invokes Earth Engine to create a buffered grid over the input feature geometry and randomly
    samples a point from each grid box

    Args:
        feature: ee.Feature object that
        gridScale: Determines the size/spacing of grid boxes
        pointScale:
        seed:

    Returns:

    """
    projection = ee.Projection("EPSG:3310").atScale(gridScale)
    geometry = feature.geometry()

    baseGrid = ee.Image.random(seed, distribution="uniform").multiply(1e6).int()
    pointsLayer = ee.Image.random(seed).multiply(1e6).int()
    mask = ee.Image.pixelCoordinates(projection
                  ).expression("!((b('x') + 0.5) % 2 != 0 || (b('y') + 0.5) % 2 != 0)")

    grid = baseGrid.clip(geometry
                  ).updateMask(mask
                  ).reproject(projection)

    gridMax = grid.addBands(pointsLayer
                 ).reduceConnectedComponents(ee.Reducer.max())

    points = pointsLayer.eq(gridMax
                       ).selfMask(
                       ).clip(geometry
                       ).reproject(projection.scale(pointScale, pointScale))

    return points.reduceToVectors(reducer=ee.Reducer.countEvery(),
                                  geometry=geometry,
                                  geometryType="centroid",
                                  maxPixels=1e10).geometry().coordinates()


def formatToGPD(fireNames, pointLst):
    """
    Pulls generated sample points from Earth Engine servers to client and formats the result as
    a GeoPandas dataframe

    Args:
        fireNames:
        pointLst:

    Returns:
        GeoPandas dataframe with sample point and corresponding fire names
    """
    names = reduce(iconcat, [[fireNames[index]]*len(value) for index, value in enumerate(pointLst)], [])
    points = reduce(iconcat, pointLst, [])
    points_gpd = gpd.GeoSeries(map(Point, points))

    gdf = gpd.GeoDataFrame(names, geometry=points_gpd).rename(columns={0: "FIRE_NAME"})
    gdf.crs = "EPSG:4326"       # for naive geometries
    gdf.to_crs("EPSG:3310", inplace=True)
    return gdf
    # gdf.to_file(path)


def mosaicByDate(collection):
    """
    Combines image tiles in a collection into a single image by day
    """
    ts = list(map(convertDate, collection.reduceColumns(reducer=ee.Reducer.toList(),
                                                        selectors=["system:time_start"]
                                        ).get('list').getInfo()))

    # unique image dates in a collection in sorted order
    ts = ee.List(list({i: None for i in ts}.keys()))

    return ts.map(lambda x: collection.filterDate(ee.Date(x),
                                                  ee.Date(x).advance(1, "day")
                                     ).mosaic(
                                     ).set("mosaicDate", x))


def pointReducer(image, collection, scale, reducer):
    """
    Takes an image, reduces over a FeatureCollection, and converts to ee.List

    Args:
        image: ee.Image to reduce
        collection: ee.FeatureCollection with geometries to reduce over
        scale: Determines nominal scale of reducer
        reducer: ee.Reducer to apply over collection

    Returns:

    """
    reducedPoints = image.reduceRegions(collection=collection,
                                        scale=scale,
                                        tileScale=2,
                                        reducer=reducer)

    return reducedPoints.toList(reducedPoints.size())


##### clean up function #####
def saveSampleData(data, keys, geometry, path):
    """

    """
    getKeys = lambda x: list(itemgetter(*keys)(x["properties"]))
    data = reduce(iconcat, data, [])

    df = pd.DataFrame(list(map(getKeys, data)))
    df["x_coord"] = list(geometry.x)
    df["y_coord"] = list(geometry.y)
    df.columns = keys + ["x_coord", "y_coord"]
    df.to_csv(path, index=False)
