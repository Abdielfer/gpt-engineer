import rasterio
from typing import List
from rasterio.enums import Resampling

class MultiLayerTiffWriter:
    @staticmethod
    def write_multilayer_tiff(file_path: str, data: List[np.ndarray], file_extension: str, crs: str) -> None:
        profile = {
            'driver': file_extension,
            'dtype': 'float32',
            'count': len(data),
            'width': data[0].shape[1],
            'height': data[0].shape[0],
            'crs': crs,
            'transform': rasterio.transform.from_origin(0, 0, 1, 1)
        }
        
        with rasterio.open(file_path, 'w', **profile) as dst:
            for i, layer in enumerate(data):
                dst.write(layer.astype('float32'), i + 1)
