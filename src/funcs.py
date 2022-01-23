import ee
import folium
import numpy as np
import urllib.request
from datetime import datetime, timedelta

import geemap


def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1, min_zoom=0):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        show=show,
        opacity=opacity,
        min_zoom=min_zoom,
        overlay=True,
        control=True
        ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer


def downloadGif(collection, gifParams, filePath, textSequence, textPosition, imageDuration):
    """
    Generates a gif using Earth Engine, downloads to directory, and adds text overlay.

    Args:
        collection: ee.ImageCollection with images to convert into gif
        gifParams:
        textSequence: Array with text to apply to each image
        textPosition: Tuple with
        imageDuration:
    """
    gifURL = collection.getVideoThumbURL(gifParams)

    urllib.request.urlretrieve(gifURL, filePath)

    geemap.add_text_to_gif(in_gif=filePath, out_gif=filePath,
                           xy=textPosition, text_sequence=textSequence,
                           duration=imageDuration, font_color="red")


def convertDate(date):
    """
    Converts EE.Date or unix date to Y-M-D formst
    """
    if isinstance(date, ee.Date):
        date = date.getInfo()["value"]

    return datetime.utcfromtimestamp(date/1000).strftime("%Y-%m-%d")# %H:%M:%S')
