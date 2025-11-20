import sys
import numpy as np

from file_handler import load_nifti, save_nifti

MEAN = 100
STD = 500


def add_noise(data: np.ndarray) -> np.ndarray:
    return data + np.random.normal(MEAN, STD, data.shape)
    #todo: add smaller data.shape and resize


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data, affine, header = load_nifti(input_path)
    data = add_noise(data)
    save_nifti(output_path, data, affine, header)
