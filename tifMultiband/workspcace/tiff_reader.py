import rasterio
import numpy as np
from typing import Tuple

class TIFFReader:
    @staticmethod
    def read_tiff(file_path: str) -> Tuple[np.ndarray, str, str]:
        try:
            with rasterio.open(file_path) as dataset:
                image_data = dataset.read()
                file_extension = dataset.profile['driver']
                crs = dataset.crs.to_string()
            return image_data, file_extension, crs
        except Exception as e:
            print(f"The TIFF in the path {file_path} is corrupted.")
            return None, None, None
