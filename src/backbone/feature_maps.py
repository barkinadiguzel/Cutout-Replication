# This module is used to extract intermediate feature maps from the backbone
# in order to analyze how Cutout affects spatial attention and feature usage.

import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
