import sys
import numpy as np
from scipy.ndimage import zoom

from file_handler import load_nifti, save_nifti

MEAN = 100
STD = 300


def create_noise(data_shape: tuple) -> np.ndarray:
    # block size determines how many pixels share similar intensity
    block_factor = 10  # increase for larger smoother regions

    # create coarse noise
    small_shape = tuple(max(1, s // block_factor) for s in data_shape)
    noise_small = np.random.normal(MEAN, STD, small_shape)

    # enlarge back (nearest → blocky structure preserved)
    zoom_factors = [data_shape[i] / small_shape[i] for i in range(len(data_shape))]
    noise_blocky = zoom(noise_small, zoom=zoom_factors, order=0)

    return noise_blocky


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data, affine, header = load_nifti(input_path)
    data = create_noise(data)
    save_nifti(output_path, data, affine, header)
