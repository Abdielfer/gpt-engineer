import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor
from metrics import Metrics

class UNetModel:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.metrics = Metrics()

    def train(self, train_data, val_data, batch_size=4, epochs=50):
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_precision = 0.0
            train_recall = 0.0
            train_iou = 0.0

            for i in range(0, len(train_data), batch_size):
                batch_images = train_data[i:i+batch_size]
                batch_masks = val_data[i:i+batch_size]

                batch_images = torch.stack([to_tensor(img) for img in batch_images]).to(self.device)
                batch_masks = torch.stack([to_tensor(mask) for mask in batch_masks]).to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_images)

                # Calculate loss
                loss = self.loss_fn(outputs, batch_masks)
                train_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                train_acc += self.metrics.calculate_accuracy(outputs, batch_masks)
                train_precision += self.metrics.calculate_precision(outputs, batch_masks)
                train_recall += self.metrics.calculate_recall(outputs, batch_masks)
                train_iou += self.metrics.calculate_iou(outputs, batch_masks)

            # Calculate average metrics
            train_loss /= len(train_data) / batch_size
            train_acc /= len(train_data) / batch_size
            train_precision /= len(train_data) / batch_size
            train_recall /= len(train_data) / batch_size
            train_iou /= len(train_data) / batch_size

            # Print metrics for the epoch
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Train Precision: {train_precision:.4f}")
            print(f"Train Recall: {train_recall:.4f}")
            print(f"Train IoU: {train_iou:.4f}")

    def evaluate(self, test_data):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_precision = 0.0
        test_recall = 0.0
        test_iou = 0.0

        with torch.no_grad():
            for i in range(len(test_data)):
                test_image = test_data[i][0]
                test_mask = test_data[i][1]

                test_image = to_tensor(test_image).unsqueeze(0).to(self.device)
                test_mask = to_tensor(test_mask).unsqueeze(0).to(self.device)

                # Forward pass
                output = self.model(test_image)

                # Calculate loss
                loss = self.loss_fn(output, test_mask)
                test_loss += loss.item()

                # Calculate metrics
                test_acc += self.metrics.calculate_accuracy(output, test_mask)
                test_precision += self.metrics.calculate_precision(output, test_mask)
                test_recall += self.metrics.calculate_recall(output, test_mask)
                test_iou += self.metrics.calculate_iou(output, test_mask)

        # Calculate average metrics
        test_loss /= len(test_data)
        test_acc /= len(test_data)
        test_precision /= len(test_data)
        test_recall /= len(test_data)
        test_iou /= len(test_data)

        # Print evaluation metrics
        print("Evaluation Metrics:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test IoU: {test_iou:.4f}")

    def save_model(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)
