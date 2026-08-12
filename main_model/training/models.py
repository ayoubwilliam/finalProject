"""
CNN model architectures for change detection.
Both models share a common pattern: encode prior and current separately,
subtract in latent space, then decode the difference to a heatmap.
"""

import torch
import torch.nn as nn

import config as cfg


class SimpleConvBlock(nn.Module):
    """Class SimpleConvBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Functionality for forward."""
        return self.conv(x)


class SimpleDiffNet(nn.Module):
    """Lightweight encoder-decoder with latent subtraction."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            SimpleConvBlock(1, 16), nn.MaxPool2d(2),
            SimpleConvBlock(16, 32), nn.MaxPool2d(2),
            SimpleConvBlock(32, 64),
        )
        self.decoder = nn.Sequential(
            SimpleConvBlock(64, 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SimpleConvBlock(64, 32), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SimpleConvBlock(32, 16), nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, prior, current):
        """Functionality for forward."""
        return self.decoder(self.encoder(current) - self.encoder(prior))


class VGGBlock(nn.Module):
    """Class VGGBlock."""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Functionality for forward."""
        return self.block(x)


class VGGDiffNet(nn.Module):
    """VGG-style encoder-decoder with 5 encoding stages and latent subtraction."""

    def __init__(self):
        super().__init__()
        self.enc1, self.pool1 = VGGBlock(1, 64, 2), nn.MaxPool2d(2, 2)
        self.enc2, self.pool2 = VGGBlock(64, 128, 2), nn.MaxPool2d(2, 2)
        self.enc3, self.pool3 = VGGBlock(128, 256, 3), nn.MaxPool2d(2, 2)
        self.enc4, self.pool4 = VGGBlock(256, 512, 3), nn.MaxPool2d(2, 2)
        self.enc5 = VGGBlock(512, 512, 3)

        self.dec5 = VGGBlock(512, 512, 3)
        self.up4, self.dec4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), VGGBlock(512, 256, 3)
        self.up3, self.dec3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), VGGBlock(256, 128, 3)
        self.up2, self.dec2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), VGGBlock(128, 64, 2)
        self.up1, self.dec1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), VGGBlock(64, 64, 2)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward_encoder(self, x):
        """Functionality for forward_encoder."""
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.pool4(self.enc4(x))
        x = self.enc5(x)
        return x

    def forward(self, prior, current):
        """Functionality for forward."""
        diff = self.forward_encoder(current) - self.forward_encoder(prior)

        x = self.dec5(diff)
        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))

        return self.final_conv(x)


def get_active_model():
    """Instantiates the model architecture specified in config.py."""
    if cfg.SELECTED_MODEL == "SimpleDiffNet":
        return SimpleDiffNet()
    if cfg.SELECTED_MODEL == "VGGDiffNet":
        return VGGDiffNet()
    raise ValueError(f"Model '{cfg.SELECTED_MODEL}' is not recognized.")
