import torch.nn as nn

def get_pooling(pool="max", kernel_size=2, stride=2):
    if pool == "max":
        return nn.MaxPool2d(kernel_size, stride)
    elif pool == "avg":
        return nn.AvgPool2d(kernel_size, stride)
    else:
        raise ValueError("Unsupported pooling")
