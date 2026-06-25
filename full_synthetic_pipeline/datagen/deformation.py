"""
BSpline deformation of 3D volumes using gryds.
Pipeline: create tube → deform with soft BSpline field → produce deformed tube + mask.

The deformation uses a VERY COARSE grid and LOW displacement amplitude to preserve
tubular topology — unlike the aggressive sphere deformation in the original pipeline.
"""

import torch
import numpy as np
import gryds

import config as cfg
from datagen.shapes import create_cylinder, apply_mask


def bspline(data, grid_density_factor, deformation_factor):
    """Applies random BSpline deformation. Handles both Tensor and ndarray input."""
    is_tensor = torch.is_tensor(data)
    if is_tensor:
        data_np = data.detach().cpu().numpy().astype(np.float32)
        original_device = data.device
    else:
        data_np = data.astype(np.float32)

    x_shape = max(1, data_np.shape[0] // grid_density_factor)
    y_shape = max(1, data_np.shape[1] // grid_density_factor)
    z_shape = max(1, data_np.shape[2] // grid_density_factor)

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


# ==============================================================================
# LEGACY — kept for backward compatibility if needed
# ==============================================================================

def get_deformed_sphere(data, intensity, pos, radius, grid_density_factor, deformation_factor):
    """Creates a full-volume deformed sphere (legacy, memory-heavy version)."""
    from datagen.shapes import create_sphere
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
    from datagen.shapes import create_sphere
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


# ==============================================================================
# NEW — ET-TUBE DEFORMATION
# ==============================================================================

def get_deformed_tube_fast(current_volume_tensor, wall_intensity, lumen_intensity,
                           pos, tube_length, outer_radius, inner_radius, margin):
    """
    Memory-efficient ET-tube generation and deformation.

    Steps:
      1. Create a small working volume just big enough for the tube + margin.
      2. Generate a hollow cylinder (wall + lumen) inside it.
      3. Apply SOFT B-spline deformation to create realistic tube curvature.
      4. Place the deformed tube into the full CT volume at position `pos`.

    The cylinder extends along dim-2 (Z) so it appears VERTICAL on frontal DRR.

    Args:
        current_volume_tensor: Full CT volume tensor (used only for shape/device).
        wall_intensity:        HU value for the radiopaque tube wall.
        lumen_intensity:       HU value for the air-filled inner channel.
        pos:                   (x, y, z) — insertion center in full-volume coordinates.
        tube_length:           Length of the tube in voxels.
        outer_radius:          Outer radius of the tube wall.
        inner_radius:          Inner radius of the tube lumen.
        margin:                Extra voxels around tube for deformation headroom.

    Returns:
        tube_vol: Full-size volume with the tube inserted (float tensor on GPU).
        mask:     Full-size boolean mask of all tube voxels.
    """
    data_shape = current_volume_tensor.shape

    # --- 1. Build the small working volume ---
    # Dims 0,1 are cross-section; dim-2 (Z) is elongated for tube length
    size_x = 2 * outer_radius + 2 * margin  # cross-section
    size_y = 2 * outer_radius + 2 * margin  # cross-section
    size_z = tube_length + 2 * margin       # along tube axis (vertical)

    center_x = size_x // 2
    center_y = size_y // 2
    center_z = size_z // 2

    # --- 2. Generate hollow cylinder primitive (CPU, numpy) ---
    small_volume = np.zeros((size_x, size_y, size_z), dtype=np.float32)
    wall_mask, lumen_mask = create_cylinder(
        small_volume,
        center=(center_x, center_y, center_z),
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        length=tube_length,
    )

    # Fill wall with high HU + noise, lumen with air HU (no noise)
    apply_mask(small_volume, wall_mask, wall_intensity, is_noised=True)
    apply_mask(small_volume, lumen_mask, lumen_intensity, is_noised=False)

    # --- 3. Apply SOFT B-spline deformation (CPU) ---
    # Uses coarse grid (TUBE_GRID_DENSITY_FACTOR=4) and tiny displacement
    # (TUBE_DEFORMATION_FACTOR=0.03) to create smooth airway-following bends
    # without collapsing the tube topology.
    deformed_small_np = bspline(
        small_volume,
        cfg.TUBE_GRID_DENSITY_FACTOR,
        cfg.TUBE_DEFORMATION_FACTOR,
    )

    # --- 4. Move to GPU ---
    deformed_small_tube = torch.from_numpy(deformed_small_np).float().to(cfg.DEVICE)

    # --- 5. Place into full-size volume ---
    tube_vol = torch.zeros_like(current_volume_tensor)

    x, y, z = pos
    half_x = size_x // 2
    half_y = size_y // 2
    half_z = size_z // 2

    # Raw destination coordinates
    x_start = int(x - half_x)
    x_end = int(x_start + size_x)
    y_start = int(y - half_y)
    y_end = int(y_start + size_y)
    z_start = int(z - half_z)
    z_end = int(z_start + size_z)

    # Clamp to volume bounds (destination)
    dst_x0 = max(0, x_start)
    dst_x1 = min(data_shape[0], x_end)
    dst_y0 = max(0, y_start)
    dst_y1 = min(data_shape[1], y_end)
    dst_z0 = max(0, z_start)
    dst_z1 = min(data_shape[2], z_end)

    # Corresponding source coordinates
    src_x0 = dst_x0 - x_start
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y0 = dst_y0 - y_start
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_z0 = dst_z0 - z_start
    src_z1 = src_z0 + (dst_z1 - dst_z0)

    # Safe assignment — only if there's a valid overlap region
    if (dst_x1 > dst_x0 and dst_y1 > dst_y0 and dst_z1 > dst_z0):
        tube_vol[dst_x0:dst_x1,
                 dst_y0:dst_y1,
                 dst_z0:dst_z1] = deformed_small_tube[src_x0:src_x1,
                                                       src_y0:src_y1,
                                                       src_z0:src_z1]

    # Build mask: any voxel with significant intensity after deformation.
    # We use a small absolute threshold to remove B-spline interpolation fuzz
    # while preserving both the high-HU wall AND the negative-HU lumen.
    mask = torch.abs(tube_vol) > 1.0

    return tube_vol, mask
