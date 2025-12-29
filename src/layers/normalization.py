import torch.nn as nn

def get_norm(channels, norm="bn"):
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    elif norm == "ln":
        return nn.GroupNorm(1, channels)
    else:
        raise ValueError("Unsupported normalization")
