import sys
import numpy as np
from scipy.ndimage import zoom

from file_handler import load_nifti, save_nifti

MEAN = 100
STD = 300
# block size determines how many pixels share similar intensity (increase for larger smoother regions)
BLOCK_FACTOR = 4


def create_noise(data_shape: tuple) -> np.ndarray:
    # create coarse noise
    small_shape = tuple(max(1, s // BLOCK_FACTOR) for s in data_shape)
    noise_small = np.random.normal(MEAN, STD, small_shape)

    # enlarge back (nearest → blocky structure preserved)
    zoom_factors = [data_shape[i] / small_shape[i] for i in range(len(data_shape))]
    noise_blocky = zoom(noise_small, zoom=zoom_factors, order=0)

    return noise_blocky