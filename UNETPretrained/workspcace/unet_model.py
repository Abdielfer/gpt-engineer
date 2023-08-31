import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub

class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.net = hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
        self.classification_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classification_head(x)
        return x
