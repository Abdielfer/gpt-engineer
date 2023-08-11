import os
from osgeo import gdal, ogr, osr

def reproject_tif(tif_file, output_crs):
    """
    Reprojects a TIFF file to the specified coordinate reference system (CRS).

    Args:
        tif_file (str): Path to the input TIFF file.
        output_crs (str): Output coordinate reference system (CRS) in the format 'EPSG:<code>'.

    Returns:
        str: Path to the reprojected TIFF file.
    """
    # Open the input TIFF file
    dataset = gdal.Open(tif_file)

    # Get the input CRS
    input_crs = dataset.GetProjection()

    # Create a spatial reference object for the input CRS
    input_srs = osr.SpatialReference()
    input_srs.ImportFromWkt(input_crs)

    # Create a spatial reference object for the output CRS
    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(int(output_crs.split(':')[1]))

    # Create a transformation object to convert from input CRS to output CRS
    transform = osr.CoordinateTransformation(input_srs, output_srs)

    # Get the output file path
    output_file = os.path.splitext(tif_file)[0] + '_reprojected.tif'

    # Create the output dataset
    output_dataset = gdal.Warp(output_file, dataset, dstSRS=output_srs, srcSRS=input_srs, resampleAlg=gdal.GRA_Bilinear)

    # Close the datasets
    dataset = None
    output_dataset = None

    return output_file

def crop_tif(tif_file, shapefile, output_file):
    """
    Crops a TIFF file using a shapefile as a mask.

    Args:
        tif_file (str): Path to the input TIFF file.
        shapefile (str): Path to the input shapefile.
        output_file (str): Path to the output TIFF file.

    Returns:
        str: Path to the output TIFF file.
    """
    # Open the input TIFF file
    dataset = gdal.Open(tif_file)

    # Open the shapefile
    shapefile_ds = ogr.Open(shapefile)
    layer = shapefile_ds.GetLayer()

    # Get the extent of the shapefile
    extent = layer.GetExtent()

    # Set the output file format
    driver = gdal.GetDriverByName('GTiff')

    # Create the output dataset
    output_dataset = driver.Create(output_file, dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount, dataset.GetRasterBand(1).DataType)

    # Set the geotransform and projection
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())

    # Set the output dataset extent to the shapefile extent
    output_dataset.SetExtent(*extent)

    # Perform the cropping
    gdal.Warp(output_dataset, dataset, cutlineDSName=shapefile, cropToCutline=True)

    # Close the datasets
    dataset = None
    output_dataset = None
    shapefile_ds = None

    return output_file
