"""
BSpline deformation of 3D volumes using gryds.
Pipeline: create sphere → deform with random BSpline field → produce deformed mass + mask.
"""

import torch
import numpy as np
import gryds

import config as cfg
from datagen.shapes import create_sphere, apply_mask


def bspline(data, grid_density_factor, deformation_factor):
    """Applies random BSpline deformation. Handles both Tensor and ndarray input."""
    is_tensor = torch.is_tensor(data)
    if is_tensor:
        data_np = data.detach().cpu().numpy().astype(np.float32)
        original_device = data.device
    else:
        data_np = data.astype(np.float32)

    x_shape = data_np.shape[0] // grid_density_factor
    y_shape = data_np.shape[1] // grid_density_factor
    z_shape = data_np.shape[2] // grid_density_factor

    gridx = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    gridy = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    gridz = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor

    # use CUDA-accelerated gryds when available
    if cfg.DEVICE_STR == "cuda":
        transform = gryds.BSplineTransformationCuda([gridx, gridy, gridz])
        interpolator = gryds.BSplineInterpolatorCuda(data_np, order=3, mode="mirror")
    else:
        transform = gryds.BSplineTransformation([gridx, gridy, gridz])
        interpolator = gryds.Interpolator(data_np, order=3, mode="mirror")

    nifti_roi = interpolator.transform(transform)

    if is_tensor:
        return torch.from_numpy(nifti_roi).to(original_device)

    return nifti_roi


def get_deformed_sphere(data, intensity, pos, radius, grid_density_factor, deformation_factor):
    """Creates a full-volume deformed sphere (legacy, memory-heavy version)."""
    sphere_np = np.zeros(data.shape, dtype=np.float32)
    sphere_mask_np = create_sphere(sphere_np, pos, radius)
    apply_mask(sphere_np, sphere_mask_np, intensity, True)

    sphere_tensor = torch.from_numpy(sphere_np).to(cfg.DEVICE)

    deformed_sphere = bspline(sphere_tensor, grid_density_factor, deformation_factor)
    mask = torch.round(deformed_sphere) != 0

    return deformed_sphere, mask


def get_deformed_sphere_fast(current_volume_tensor, intensity, pos, radius, margin,
                             grid_density_factor, deformation_factor):
    """Memory-efficient deformation: generates a small sphere, deforms it, places into full volume."""
    data_shape = current_volume_tensor.shape

    size = 2 * radius + margin
    center = size // 2

    # 1. generate small sphere (CPU)
    small_sphere_volume = np.zeros((size, size, size), dtype=np.float32)
    sphere_mask_np = create_sphere(small_sphere_volume, [center, center, center], radius)
    apply_mask(small_sphere_volume, sphere_mask_np, intensity, is_noised=True)

    # 2. deform small sphere (CPU)
    deformed_small_np = bspline(small_sphere_volume, grid_density_factor, deformation_factor)

    # 3. move to GPU
    deformed_small_sphere = torch.from_numpy(deformed_small_np).float().to(cfg.DEVICE)

    # 4. place into full-size volume
    sphere_vol = torch.zeros_like(current_volume_tensor)

    x, y, z = pos
    half_size = size // 2

    # raw coordinates
    x_start = int(x - half_size)
    x_end = int(x_start + size)
    y_start = int(y - half_size)
    y_end = int(y_start + size)
    z_start = int(z - half_size)
    z_end = int(z_start + size)

    # clamp destination coordinates
    sphere_x_start = max(0, x_start)
    sphere_x_end = min(data_shape[0], x_end)
    sphere_y_start = max(0, y_start)
    sphere_y_end = min(data_shape[1], y_end)
    sphere_z_start = max(0, z_start)
    sphere_z_end = min(data_shape[2], z_end)

    # source coordinates
    small_x_start = sphere_x_start - x_start
    small_x_end = small_x_start + (sphere_x_end - sphere_x_start)
    small_y_start = sphere_y_start - y_start
    small_y_end = small_y_start + (sphere_y_end - sphere_y_start)
    small_z_start = sphere_z_start - z_start
    small_z_end = small_z_start + (sphere_z_end - sphere_z_start)

    # safe assignment
    if (sphere_x_end > sphere_x_start and
            sphere_y_end > sphere_y_start and
            sphere_z_end > sphere_z_start):
        sphere_vol[sphere_x_start:sphere_x_end,
        sphere_y_start:sphere_y_end,
        sphere_z_start:sphere_z_end] = deformed_small_sphere[
            small_x_start:small_x_end,
            small_y_start:small_y_end,
            small_z_start:small_z_end]

    # remove bspline interpolation fuzz
    mask = torch.round(sphere_vol) != 0

    return sphere_vol, mask
