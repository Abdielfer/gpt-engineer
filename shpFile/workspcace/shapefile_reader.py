import os
from typing import Tuple
from osgeo import ogr

def get_shapefile_bounding_box(shapefile_path: str) -> Tuple[float, float, float, float]:
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shape file not found at path: {shapefile_path}")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = driver.Open(shapefile_path, 0)
    if shapefile is None:
        raise ValueError(f"Failed to open shape file at path: {shapefile_path}")

    layer = shapefile.GetLayer()
    extent = layer.GetExtent()

    shapefile.Destroy()

    return extent
