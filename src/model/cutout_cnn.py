import torch.nn as nn
from backbone.cnn_blocks import SimpleCNN
from cutout.cutout_layer import Cutout

class CutoutCNN(nn.Module):
    def __init__(self, num_classes=10, cutout_size=16):
        super().__init__()
        self.cutout = Cutout(cutout_size)
        self.backbone = SimpleCNN()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, apply_cutout=False):
        if self.training and apply_cutout:
            for i in range(x.size(0)):
                x[i] = self.cutout(x[i])

        feat = self.backbone(x)
        feat = self.pool(feat).squeeze(-1).squeeze(-1)
        return self.fc(feat)
