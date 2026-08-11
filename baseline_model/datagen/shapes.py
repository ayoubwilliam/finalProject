"""
3D geometric shape primitives (sphere, ellipsoid, box) and mask application.
Used to create the base object before deformation.
"""

import numpy as np

from datagen.noise import create_noise


def create_box(data, ranges):
    """Creates a boolean box mask within the given axis ranges."""
    x0, x1 = ranges[0]
    y0, y1 = ranges[1]
    z0, z1 = ranges[2]

    mask = np.zeros_like(data, dtype=bool)
    mask[x0:x1, y0:y1, z0:z1] = True
    return mask


def create_sphere(data, center, radius):
    """Creates a boolean sphere mask centered at (cx, cy, cz) with given radius."""
    cx, cy, cz = center

    x = np.arange(0, data.shape[0])[:, None, None]
    y = np.arange(0, data.shape[1])[None, :, None]
    z = np.arange(0, data.shape[2])[None, None, :]

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

    mask = dist <= radius
    return mask


def create_ellipsoid(data, center, radius):
    """Creates a boolean ellipsoid mask with per-axis radii."""
    cx, cy, cz = center
    rx, ry, rz = radius

    x = np.arange(data.shape[0])[:, None, None]
    y = np.arange(data.shape[1])[None, :, None]
    z = np.arange(data.shape[2])[None, None, :]

    dist = np.sqrt(((x - cx) / rx) ** 2 +
                   ((y - cy) / ry) ** 2 +
                   ((z - cz) / rz) ** 2)

    mask = dist <= 1
    return mask


def apply_mask(data, mask, intensity, is_noised):
    """Fills masked voxels with intensity, optionally adding blocky noise."""
    if is_noised:
        noise = create_noise(data.shape)
        data[mask] = intensity + noise[mask]
    else:
        data[mask] = intensity

    return data
