import torch.nn as nn
from layers.conv_block import ConvBlock

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, channels=64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, channels),
            ConvBlock(channels, channels),
            nn.MaxPool2d(2),

            ConvBlock(channels, channels * 2),
            ConvBlock(channels * 2, channels * 2),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.features(x)
