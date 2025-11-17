import sys
import numpy as np

from file_handler import load_nifti, save_nifti


def create_box(data: np.ndarray, ranges: list[list], intensity: int) -> np.ndarray:
    """
        Sets all voxels inside a 3D box to a given intensity.
        ranges: [[x_start, x_end], [y_start, y_end], [z_start, z_end]].
        data: 3D numpy array of the CT volume.
        intensity: value to assign inside the box.
        Returns the modified array.
    """
    for i in range(ranges[0][0], ranges[0][1]):
        for j in range(ranges[1][0], ranges[1][1]):
            for k in range(ranges[2][0], ranges[2][1]):
                data[i, j, k] = intensity
    return data


def create_sphere(data: np.ndarray, center: list[int, int, int], radius: int, intensity: int) -> np.ndarray:
    """
        Sets all voxels inside a spherical region to a given intensity.
        center: [x, y, z] voxel coordinates of sphere center.
        radius: sphere radius in voxels.
        data: 3D numpy array of the CT volume.
        intensity: value to assign inside the sphere.
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
    data[dist <= radius] = intensity
    return data


def create_ellipsoid(data: np.ndarray, center: list[int], radius: list[int],
                     intensity: int) -> np.ndarray:
    """
    Sets all voxels inside an ellipsoid to a given intensity.
    center: [cx, cy, cz] voxel coordinates of ellipsoid center.
    radius: [rx, ry, rz] ellipsoid radii in x, y, z directions.
    data: 3D numpy array of the CT volume.
    intensity: value to assign inside the ellipsoid.
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
    data[dist <= 1] = intensity
    return data


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    intensity = int(sys.argv[3])

    data, affine, header = load_nifti(input_path)

    # cube = create_box(data, [[153, 213], [223, 283], [266, 326]], intensity)
    # box = create_box(data, [[153, 213], [223, 283], [246, 346]], intensity)
    # sphere = create_sphere(data, [180, 250, 300], 30, intensity)
    # sphere = create_ellipsoid(data, [180, 250, 300], [30, 40, 60], intensity)

    save_nifti(output_path, data, affine, header)
