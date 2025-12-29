import torch

def normalize(img, mean, std):
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    return img
