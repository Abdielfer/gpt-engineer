import gdal_utils

def reproject_crop_tif(tif_file, shapefile, output_file):
    """
    Reprojects and crops a TIFF file using a shapefile as a mask.

    Args:
        tif_file (str): Path to the input TIFF file.
        shapefile (str): Path to the input shapefile.
        output_file (str): Path to the output TIFF file.

    Returns:
        str: Path to the output TIFF file.
    """
    # Reproject the TIFF file
    reprojected_file = gdal_utils.reproject_tif(tif_file, output_crs='EPSG:4326')

    # Crop the reprojected TIFF file using the shapefile as a mask
    cropped_file = gdal_utils.crop_tif(reprojected_file, shapefile, output_file)

    return cropped_file
