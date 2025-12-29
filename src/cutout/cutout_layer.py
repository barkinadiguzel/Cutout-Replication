import torch
import random

class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        _, h, w = img.shape
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        half = self.size // 2
        y1 = max(0, y - half)
        y2 = min(h, y + half)
        x1 = max(0, x - half)
        x2 = min(w, x + half)

        img[:, y1:y2, x1:x2] = 0.0
        return img
