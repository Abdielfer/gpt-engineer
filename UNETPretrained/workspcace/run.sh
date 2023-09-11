# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install the required packages
pip install torch torchvision pillow

# Run the entrypoint.py script
python entrypoint.py
