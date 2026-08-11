"""
CNN model architectures for change detection.
Both models share a common pattern: encode prior and current separately,
subtract in latent space, then decode the difference to a heatmap.
"""

import torch
import torch.nn as nn

import config as cfg


class AngleRegressionNet(nn.Module):
    """
    CNN that takes prior and current DRRs concatenated channel-wise (2 channels)
    and predicts a single scalar value representing the 2D in-plane rotation angle.
    """

    def __init__(self):
        super().__init__()
        # Input is 2 channels (prior, current)
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, stride=2), # 256x256
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1) # Single scalar output (angle)
        )

    def forward(self, prior, current):
        # Concatenate prior and current along channel dim
        x = torch.cat([prior, current], dim=1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        angle = self.regressor(x)
        return angle.squeeze(1)


def get_active_model():
    """Instantiates the AngleRegressionNet baseline."""
    return AngleRegressionNet()
