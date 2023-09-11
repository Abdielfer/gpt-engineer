import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.hub import load
from tqdm import tqdm
from PIL import Image

# Define the UNet model with classification head
class UNetWithClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(UNetWithClassifier, self).__init__()
        self.net = load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=pretrained, scale=0.5)
        self.classifier = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.net(x)
        output = self.classifier(features)
        return output

# Custom Dataset class for reading GeoTIFF images
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Set random seed for reproducibility
torch.manual_seed(0)

# Define data transformations and create DataLoader for training and validation
transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90, p=0.5),
    transforms.ToTensor()
])

train_dataset = CustomDataset("UNETPretrained/input", transform=transformations)
val_dataset = CustomDataset("UNETPretrained/validation", transform=transforms.ToTensor())

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the UNet model with classification head
model = UNetWithClassifier(pretrained=True, num_classes=1)
model.apply(lambda x: nn.init.normal_(x.weight, mean=0, std=0.01))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

# Training loop
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images in tqdm(train_loader):
        images = images.to(device)
        labels = torch.Tensor(np.random.choice([0, 1], size=images.shape[0])).to(device)  # Replace with actual labels
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            labels = torch.Tensor(np.random.choice([0, 1], size=images.shape[0])).to(device)  # Replace with actual labels
            
            outputs = model(images)
            val_loss += criterion(outputs.view(-1), labels.view(-1)).item()
    
    val_loss /= len(val_loader)
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}")
    
    # Save the model if it has the best validation loss so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'UNETPretrained/model/best_model.pth')

# Evaluation on an independent dataset
# Load the saved best model
model.load_state_dict(torch.load('UNETPretrained/model/best_model.pth'))
model.eval()

# Define a function to evaluate the model on the test dataset
def evaluate_model(model, test_dataset):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images in tqdm(test_dataset):
            images = images.unsqueeze(0).to(device)  # Batch size of 1
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())
            true_labels.append(np.random.choice([0, 1]))  # Replace with actual labels

    predictions = np.vstack(predictions)
    true_labels = np.array(true_labels)

    accuracy = accuracy_score(true_labels, (predictions >= 0.5).astype(int))
    precision = precision_score(true_labels, (predictions >= 0.5).astype(int))
    recall = recall_score(true_labels, (predictions >= 0.5).astype(int))
    f1 = f1_score(true_labels, (predictions >= 0.5).astype(int))
    iou = jaccard_score(true_labels, (predictions >= 0.5).astype(int))

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"IoU: {iou}")

# Load the test dataset and evaluate the model
test_dataset = CustomDataset("UNETPretrained/test", transform=transforms.ToTensor())
evaluate_model(model, test_dataset)
