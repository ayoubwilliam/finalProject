"""
Blocky noise generation for adding texture inside synthetic masses.
Generates coarse random noise and upsamples to produce block-structured patterns.
"""

import numpy as np
from scipy.ndimage import zoom

import config as cfg


def create_noise(data_shape):
    """Creates blocky Gaussian noise by generating at coarse resolution and upsampling."""
    small_shape = tuple(max(1, s // cfg.NOISE_BLOCK_FACTOR) for s in data_shape)
    noise_small = np.random.normal(cfg.NOISE_MEAN, cfg.NOISE_STD, small_shape)

    # nearest-neighbor upsampling preserves block structure
    zoom_factors = [data_shape[i] / small_shape[i] for i in range(len(data_shape))]
    noise_blocky = zoom(noise_small, zoom=zoom_factors, order=0)

    return noise_blocky
