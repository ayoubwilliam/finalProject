import numpy as np
import gryds

from create_shapes import create_sphere, apply_mask


def bspline(data, grid_density_factor, deformation_factor):
    data = data.astype(np.float32)
    x_shape = data.shape[0] // grid_density_factor
    y_shape = data.shape[1] // grid_density_factor
    z_shape = data.shape[2] // grid_density_factor
    gridx = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    gridy = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    gridz = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor

    transform = gryds.BSplineTransformation([gridx, gridy, gridz])

    interpolator = gryds.Interpolator(data, order=3, mode="mirror")

    nifti_roi = interpolator.transform(transform)
    return nifti_roi


def get_deformed_sphere(data, intensity, pos, radius, grid_density_factor, deformation_factor):
    sphere = np.zeros_like(data, dtype=np.float32)
    sphere_mask = create_sphere(sphere, pos, radius)
    apply_mask(sphere, sphere_mask, intensity, True)

    deformed_sphere = bspline(sphere, grid_density_factor, deformation_factor)

    mask = np.round(deformed_sphere) != 0

    return deformed_sphere, mask


import numpy as np


def get_deformed_sphere_fast(data_shape, intensity, pos, radius, margin, grid_density_factor,
                             deformation_factor):
    size = 2 * radius + margin
    small_sphere_volume = np.zeros((size, size, size), dtype=np.float32)
    center = size // 2

    sphere_mask = create_sphere(small_sphere_volume, [center, center, center], radius)
    apply_mask(small_sphere_volume, sphere_mask, intensity, True)

    deformed_small_sphere = bspline(small_sphere_volume, grid_density_factor, deformation_factor)

    sphere = np.zeros(data_shape, dtype=np.float32)

    x, y, z = pos
    half_size = size // 2

    # --- 1. Calculate raw coordinates (might be out of bounds) ---
    x_start = x - half_size
    x_end = x_start + size
    y_start = y - half_size
    y_end = y_start + size
    z_start = z - half_size
    z_end = z_start + size

    # --- 2. Clamp coordinates to fit within the main volume (Destination) ---
    # max(0, ...) handles the left/top edges
    # min(limit, ...) handles the right/bottom edges
    sphere_x_start = max(0, x_start)
    sphere_x_end = min(data_shape[0], x_end)
    sphere_y_start = max(0, y_start)
    sphere_y_end = min(data_shape[1], y_end)
    sphere_z_start = max(0, z_start)
    sphere_z_end = min(data_shape[2], z_end)

    # --- 3. Calculate corresponding coordinates for the small sphere (Source) ---
    # We essentially crop the small sphere by the same amount we cropped the destination
    small_x_start = sphere_x_start - x_start
    small_x_end = small_x_start + (sphere_x_end - sphere_x_start)

    small_y_start = sphere_y_start - y_start
    small_y_end = small_y_start + (sphere_y_end - sphere_y_start)

    small_z_start = sphere_z_start - z_start
    small_z_end = small_z_start + (sphere_z_end - sphere_z_start)

    # --- 4. Apply the safe assignment ---
    # Only assign if the computed ranges are valid (i.e., volume is not completely off-screen)
    if (sphere_x_end > sphere_x_start and
            sphere_y_end > sphere_y_start and
            sphere_z_end > sphere_z_start):
        sphere[sphere_x_start:sphere_x_end,
        sphere_y_start:sphere_y_end,
        sphere_z_start:sphere_z_end] = deformed_small_sphere[
            small_x_start:small_x_end,
            small_y_start:small_y_end,
            small_z_start:small_z_end]

    mask = np.round(sphere) != 0

    return sphere, mask
