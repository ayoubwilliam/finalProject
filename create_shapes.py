import sys
import numpy as np

from file_handler import load_nifti, save_nifti
from noise import create_noise


def create_box(data: np.ndarray, ranges: list[list]) -> np.ndarray:
    x0, x1 = ranges[0]
    y0, y1 = ranges[1]
    z0, z1 = ranges[2]

    mask = np.zeros_like(data, dtype=bool)
    mask[x0:x1, y0:y1, z0:z1] = True

    return mask


def create_sphere(data: np.ndarray, center: list[int, int, int], radius: int) -> np.ndarray:
    cx, cy, cz = center

    # Create coordinate grids
    x = np.arange(0, data.shape[0])[:, None, None]
    y = np.arange(0, data.shape[1])[None, :, None]
    z = np.arange(0, data.shape[2])[None, None, :]

    # Compute squared distance from center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

    # Assign intensity where inside the sphere
    mask = dist <= radius
    return mask


def create_ellipsoid(data: np.ndarray, center: list[int], radius: list[int]) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radius

    # Coordinate grids
    x = np.arange(data.shape[0])[:, None, None]
    y = np.arange(data.shape[1])[None, :, None]
    z = np.arange(data.shape[2])[None, None, :]

    # Ellipsoid equation (normalized distance)
    dist = np.sqrt(((x - cx) / rx) ** 2 +
                   ((y - cy) / ry) ** 2 +
                   ((z - cz) / rz) ** 2)

    # Fill ellipsoid
    mask = dist <= 1
    return mask

def apply_mask(data: np.ndarray, mask: np.ndarray, intensity: int, is_noised:bool) -> np.ndarray:
    if is_noised:
        noise = create_noise(data.shape)
        data[mask] = intensity + noise[mask]
    else:
        data[mask] = intensity

    return data


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    intensity = int(sys.argv[3])

    data, affine, header = load_nifti(input_path)

    # create_sphere(data, [180, 250, 300], 30)
    # create_ellipsoid(data, [180, 250, 300], [30, 40, 60])
    mask = create_box(data, [[153, 213], [223, 283], [266, 326]])  # cube
    # create_box(data, [[153, 213], [223, 283], [246, 346]]) # box

    data = apply_mask(data, mask, intensity, True)

    save_nifti(output_path, data, affine, header)
