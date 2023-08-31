import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from unet_model import UNetModel
from data_loader import CustomDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
batch_size = 4
epochs = 50
learning_rate = 0.0001
validation_split = 0.1

# Define data transformations
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
])

# Create train and validation datasets
train_dataset = CustomDataset("UNETPretrained/input/train", "UNETPretrained/output/train", transform=data_transforms)
val_dataset = CustomDataset("UNETPretrained/input/val", "UNETPretrained/output/val")

# Create train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the UNet model
model = UNetModel().to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_masks)

    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "UNETPretrained/model/unet_model.pth")
