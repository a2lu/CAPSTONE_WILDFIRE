import ee
import geemap
import folium
import os
import urllib.request
import glob
from PIL import Image, ImageOps



def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1, min_zoom=0):
    """
    Maps EE objects in Folium
    """
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


def downloadEEGif(collection, gifParams, filePath, textSequence, textPosition, imageDuration):
    """
    Generates a gif using Earth Engine, downloads to directory, and adds text overlay.

    Args:
        collection: ee.ImageCollection with images to convert into gif
        gifParams:
        textSequence: array with text to apply to each image
        textPosition: tuple with text position
        imageDuration: length of each image frame in milliseconds
    """
    gifURL = collection.getVideoThumbURL(gifParams)

    urllib.request.urlretrieve(gifURL, filePath)

    geemap.add_text_to_gif(in_gif=filePath, out_gif=filePath,
                           xy=textPosition, text_sequence=textSequence,
                           duration=imageDuration, font_color="red")


def saveImage(image, params, options, path, fileName):
    """
    Saves ee.Image locally as png
    """
    dim, region = options
    params["dimensions"] = dim
    params["region"] = region

    if not os.path.exists(path):
        os.makedirs(path)

    savePath = os.path.join(path, fileName)
    urllib.request.urlretrieve(image.getThumbURL(params=params),
                               savePath)


def stdImageSize(in_path, stdSize, out_path):
    """
    Standardizes image sizes with buffers

    Args:
        in_path: path with images to standardize
        stdSize: size in pixels to standardize images
        out_path: path to store standardized images
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in sorted(glob.glob(os.path.join(in_path, "*.png"))):
        img = Image.open(i)
        x, y = img.size
        sideBuffer, topBuffer = int((stdSize-x)/2), int((stdSize-y)/2)
        border = (sideBuffer, topBuffer+10, sideBuffer, topBuffer+30)
        img_with_border = ImageOps.expand(img,border=border,fill='black')
        img_with_border.save(out_path+"/{}".format(i.split("/")[-1]))


def makeGif(in_path, out_path, text, duration, pos=("40%","96%"), size=16, color="red"):
    """
    Creates a gif with text for given set of images
    """
    path, fname = os.path.split(out_path)
    if not os.path.exists(path):
        os.makedirs(path)

    geemap.png_to_gif(in_path, out_path, fps=1, loop=0)
    geemap.add_text_to_gif(in_gif=out_path,
                           out_gif=out_path,
                           text_sequence=text,
                           duration=duration,
                           xy=pos,
                           font_size=size,
                           font_color=color)
