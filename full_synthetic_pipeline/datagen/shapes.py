"""
3D geometric shape primitives (sphere, ellipsoid, box, cylinder) and mask application.
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


def create_cylinder(data, center, outer_radius, inner_radius, length):
    """
    Creates a hollow cylinder (tube) mask along the Z-axis (dim-2).

    The tube is centered at (cx, cy, cz) and extends along dim-2 — the
    superior-inferior direction. On a frontal DRR (projected along dim-1),
    this makes the tube appear VERTICAL, matching real ET-tube anatomy.

    Args:
        data:          3D numpy array whose shape defines the coordinate grid.
        center:        (cx, cy, cz) — center of the cylinder.
        outer_radius:  Outer wall radius in voxels.
        inner_radius:  Inner lumen radius in voxels (0 = solid cylinder).
        length:        Total length of the cylinder in voxels along dim-2.

    Returns:
        wall_mask:  Boolean mask of the radiopaque tube wall.
        lumen_mask: Boolean mask of the air-filled inner lumen.
    """
    cx, cy, cz = center
    half_len = length // 2

    x = np.arange(data.shape[0])[:, None, None]
    y = np.arange(data.shape[1])[None, :, None]
    z = np.arange(data.shape[2])[None, None, :]

    # Radial distance from tube axis (axis runs along dim-2 through cx, cy)
    radial_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Axial extent: tube spans [cz - half_len, cz + half_len] along dim-2
    axial_mask = (z >= (cz - half_len)) & (z <= (cz + half_len))

    # Full cylinder (everything inside outer radius AND within axial extent)
    outer_mask = (radial_dist <= outer_radius) & axial_mask

    # Inner lumen (air channel)
    if inner_radius > 0:
        lumen_mask = (radial_dist <= inner_radius) & axial_mask
    else:
        lumen_mask = np.zeros_like(data, dtype=bool)

    # Wall = outer minus lumen
    wall_mask = outer_mask & ~lumen_mask

    return wall_mask, lumen_mask


def apply_mask(data, mask, intensity, is_noised):
    """Fills masked voxels with intensity, optionally adding blocky noise."""
    if is_noised:
        noise = create_noise(data.shape)
        data[mask] = intensity + noise[mask]
    else:
        data[mask] = intensity

    return data
