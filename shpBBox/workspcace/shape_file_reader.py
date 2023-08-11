from typing import Tuple
from osgeo import ogr

class ShapeFileReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        data_source = driver.Open(self.file_path, 0)
        layer = data_source.GetLayer()

        extent = layer.GetExtent()
        min_x, max_x, min_y, max_y = extent

        return min_x, min_y, max_x, max_y
