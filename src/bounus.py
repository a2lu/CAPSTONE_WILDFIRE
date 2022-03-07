import ee
import geopandas as gpd
import geemap #
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import os #
from datetime import datetime, timedelta
from functools import reduce
from operator import iconcat, itemgetter


def boundsBuffer(x, buffer=0.075):
    """
    Creates a bounding box with a default 7.5% buffer

    Args:
        x: GeoPandas geometry object

    Returns:
        Shapely.Polygon that represents buffered bounding box over input geometry
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
    """
    Maps NWCG fire size code based on acres burned by a fire
    """
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



def genSamplePoints(collection, fireName, gridScale, pointScale, seed):
    """
    Invokes Earth Engine to create a buffered grid over the input feature geometry and randomly
    samples a point from each grid box

    Args:
        feature: ee.Feature object that
        gridScale: Determines the size/spacing of grid boxes
        pointScale:
        seed: random seed

    Returns:
        ee.List with sample pixel coordinates
    """
    feature = collection.filter(ee.Filter.eq("FIRE_NAME", fireName))
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

    return points.reduceToVectors(reducer=ee.Reducer.countEvery(),   #ee.List
                                  geometry=geometry,
                                  geometryType="centroid",
                                  maxPixels=1e10).geometry().coordinates()

    # points = points.reduceToVectors(reducer=ee.Reducer.countEvery(),
    #                               geometry=geometry,
    #                               geometryType="centroid",
    #                               maxPixels=1e10
    #               ).geometry(
    #               ).coordinates(
    #               ).map(lambda x: ee.Feature(ee.Geometry.Point(x),
    #                                          {"FIRE_NAME": fireName}))
    #
    # return ee.FeatureCollection(points).set("FIRE_NAME", fireName)


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
    # gdf.to_crs("EPSG:3310", inplace=True)
    return gdf


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
        ee.List with sampled data from image
    """
    reducedPoints = image.reduceRegions(collection=collection,
                                        scale=scale,
                                        tileScale=2,
                                        reducer=reducer)

    return reducedPoints.toList(reducedPoints.size())


def saveSampleData(data, keys, geometry, path):
    """

    """
    getKeys = lambda x: list(itemgetter(*keys)(x["properties"]))
    data = reduce(iconcat, data, [])

    df = pd.DataFrame(list(map(getKeys, data)))
    intCols = keys[1:12] + keys[-2:]
    df["x_coord"] = list(geometry.x)
    df["y_coord"] = list(geometry.y)
    df.columns = keys + ["x_coord", "y_coord"]
    df = df.dropna().round(2)
    df[intCols] = df[intCols].astype(int)

    df.to_csv(path, index=False)


def prepImage(preFireImage, postFireImage, fireName, geometry, endDate):
    """


    """
    # Calculate NBR, dNBR, and burn severity
    preFireNBR = preFireImage.normalizedDifference(['SR_B5', 'SR_B7'])
    postFireNBR = postFireImage.normalizedDifference(['SR_B5', 'SR_B7'])
    dNBR = (preFireNBR.subtract(postFireNBR)
                     ).multiply(1000
                     ).rename("dNBR")

    burnSeverity = dNBR.expression(" (b('dNBR') > 425) ? 5 "    # purple: high severity
                                   ":(b('dNBR') > 225) ? 4 "    # orange: moderate severity
                                   ":(b('dNBR') > 100) ? 3 "    # yellow: low severity
                                   ":(b('dNBR') > -60) ? 2 "    # green: unburned/unchanged
                                   ":(b('dNBR') <= -60) ? 1 "   # brown: vegetation growth
                                   ":0"                         # pseudo mask
                      ).rename("burnSeverity")

    # Get SRTM elevation, NLCD land coverpostFireImageNDVI, and GRIDMET weather
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2016_REL'
            ).filter(ee.Filter.eq('system:index', '2016')).first()

    lc = nlcd.select("landcover"
            ).expression(" (b('landcover') > 90) ? 1 "    # blue: other (wetland)
                         ":(b('landcover') > 80) ? 6 "    # brown: agriculture
                         ":(b('landcover') > 70) ? 5 "    # lightGreen: grassland/herbaceous
                         ":(b('landcover') > 50) ? 4 "    # yellow: shrub
                         ":(b('landcover') > 40) ? 3 "    # green: forest
                         ":(b('landcover') > 30) ? 1 "    # blue: other (barren land)
                         ":(b('landcover') > 20) ? 2 "    # red: developed/urban
                         ":(b('landcover') > 10) ? 1 "    # blue: other (water/perennial ice+snow)
                         ":0"                             # handle for potential exceptions
            ).rename("landCover")

    ndvi = postFireImage.normalizedDifference(["SR_B5", "SR_B4"]
                       ).rename("NDVI"
                       ).multiply(1000)

    gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET"
               ).filterBounds(geometry
               ).filterDate(ee.Date(endDate).advance(-3, "day"), endDate
               ).mean()

    # Merge all image bands together
    combined = postFireImage.select('SR_B.'                 # post-fire L8 bands 1-7
                           ).addBands(burnSeverity          # classified burn severity
                           ).addBands(dNBR                  # dNBR
                           ).addBands(ndvi                  # post-fire NDVI
                           ).addBands(dem                   # SRTM elevation
                           ).addBands(gridmet               # all GRIDMET bands
                           ).addBands(nlcd.select("percent_tree_cover")
                           ).addBands(lc                    # simplfied landcover
                           ).set("FIRE_NAME", fireName)
    return combined


def loadTif(numTries, imgScale, fireName, image, geometry, path="tifs"):
    """
    Downloads an ee.Image as a raster at varying spatial resolutions if download
    """
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(numTries):
        try:
            geemap.ee_export_image(ee_object=image,
                                   filename=os.path.join(path, "{}.tif".format(fireName)),
                                   scale=imgScale[i],
                                   region=geometry)
            # os.path.join(path, "{}.tif".format(fireID))
            # geemap.ee_export_image(image, "tifs/{}.tif".format(fireID), scale=imgScale[i], region=geometry)
            print("Downloaded {} at {}m scale".format(fireName, imgScale[i]))
            break
        except Exception:
            if i == numTries-1:
                print("Fire exceeds total request size")
            # else:
                # print("Retrying at {}m scale".format(imgScale[i+1]))
    print("\n")
