import torch
from datetime import datetime
from unet_model import UNetModel
from data_loader import DataLoader

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset directory
dataset_dir = "UNETPretrained"

# Initialize the DataLoader
data_loader = DataLoader(dataset_dir)

# Load the input and output images
input_images, output_masks = data_loader.load_data()

# Preprocess the input and output images
preprocessed_images, preprocessed_masks = data_loader.preprocess_data(input_images, output_masks)

# Split the dataset into training and validation sets
train_data, val_data = data_loader.split_data(preprocessed_images, preprocessed_masks)

# Initialize the UNet model
model = UNetModel(device=device)

# Train the model
model.train(train_data, val_data)

# Evaluate the model
model.evaluate(val_data)

# Save the model
model.save_model("Model_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
