# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install PyTorch and torchvision
pip install torch torchvision

# Install torch-hub
pip install torch-hub

# Run the entrypoint.py file
python entrypoint.py
