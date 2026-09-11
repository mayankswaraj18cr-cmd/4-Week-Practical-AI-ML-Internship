"""Small CNN suitable for image classification and feature extraction."""

import torch
from torch import Tensor, nn


class CNNImageClassifier(nn.Module):
    """Convolutional classifier with a public feature extractor."""

    def __init__(self, num_classes: int = 10, input_channels: int = 1) -> None:
        super().__init__()
        if num_classes < 2 or input_channels < 1:
            raise ValueError("num_classes must be >= 2 and input_channels must be positive")
        self.features = nn.Sequential(nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(64, num_classes)

    def extract_features(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        return self.features(images).flatten(1)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.extract_features(images))
