# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install GDAL and OGR dependencies
sudo apt-get install libgdal-dev

# Install Python GDAL package
pip install GDAL

# Install other dependencies
pip install numpy
pip install osgeo

# Run the reproject_crop.py script
python reproject_crop.py input.tif shapefile.shp output.tif
