import sys
import numpy as np

from file_handler import load_nifti, save_nifti
from noise import add_noise


def create_box(data: np.ndarray, ranges: list[list], intensity: int, is_noised: bool = False) -> np.ndarray:
    """
        Sets all voxels inside a 3D box to a given intensity.
        ranges: [[x_start, x_end], [y_start, y_end], [z_start, z_end]].
        data: 3D numpy array of the CT volume.
        intensity: value to assign inside the box.
        is_noised: add noise only inside the box region if True
        Returns the modified array.
    """
    x0, x1 = ranges[0]
    y0, y1 = ranges[1]
    z0, z1 = ranges[2]

    # Boolean mask instead of float/int
    mask = np.zeros_like(data, dtype=bool)
    mask[x0:x1, y0:y1, z0:z1] = True

    # Set base intensity
    data[mask] = intensity

    if is_noised:
        data[mask] = add_noise(data[mask])

    return data

def create_noised_box(data: np.ndarray, ranges: list[list], intensity: int, is_noised: bool = False) -> (
        np.ndarray):
    x0, x1 = ranges[0]
    y0, y1 = ranges[1]
    z0, z1 = ranges[2]

    mask = np.zeros_like(data, dtype=bool)
    mask[x0:x1, y0:y1, z0:z1] = True

    if is_noised:
        # generate 3D noise for the whole volume, then apply only in the box
        noise_vol = add_noise(np.zeros_like(data))
        data[mask] = intensity + noise_vol[mask]
    else:
        data[mask] = intensity

    return data


def create_sphere(data: np.ndarray, center: list[int, int, int], radius: int, intensity: int,
                  is_noised: bool = False) -> np.ndarray:
    """
        Sets all voxels inside a spherical region to a given intensity.
        center: [x, y, z] voxel coordinates of sphere center.
        radius: sphere radius in voxels.
        data: 3D numpy array of the CT volume.
        intensity: value to assign inside the sphere.
        is_noised: add noise only inside the box region if True
        Returns the modified array.
    """
    cx, cy, cz = center

    # Create coordinate grids
    x = np.arange(0, data.shape[0])[:, None, None]
    y = np.arange(0, data.shape[1])[None, :, None]
    z = np.arange(0, data.shape[2])[None, None, :]

    # Compute squared distance from center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

    # Assign intensity where inside the sphere
    mask = dist <= radius
    data[mask] = intensity

    if is_noised:
        data[mask] = add_noise(data[mask])

    return data


def create_ellipsoid(data: np.ndarray, center: list[int], radius: list[int],
                     intensity: int, is_noised: bool = False) -> np.ndarray:
    """
    Sets all voxels inside an ellipsoid to a given intensity.
    center: [cx, cy, cz] voxel coordinates of ellipsoid center.
    radius: [rx, ry, rz] ellipsoid radii in x, y, z directions.
    data: 3D numpy array of the CT volume.
    intensity: value to assign inside the ellipsoid.
    is_noised: add noise only inside the box region if True
    Returns the modified volume.
    """
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
    data[mask] = intensity

    if is_noised:
        data[mask] = add_noise(data[mask])

    return data


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    intensity = int(sys.argv[3])

    data, affine, header = load_nifti(input_path)

    # create_sphere(data, [180, 250, 300], 30, intensity, True)
    # create_ellipsoid(data, [180, 250, 300], [30, 40, 60], intensity, True)
    create_noised_box(data, [[153, 213], [223, 283], [266, 326]], intensity, True)  # cube
    # create_box(data, [[153, 213], [223, 283], [246, 346]], intensity) # box

    save_nifti(output_path, data, affine, header)
