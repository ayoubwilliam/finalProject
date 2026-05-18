import torch
import numpy as np
import gryds

from create_shapes import create_cylinder, create_torus_segment
from device_constants import DEVICE


def bspline(data, grid_density_factor, deformation_factor):
    # --- Bridge: Handle PyTorch Tensor Input ---
    is_tensor = torch.is_tensor(data)
    if is_tensor:
        # gryds expects numpy, so we move to CPU and convert
        data_np = data.detach().cpu().numpy().astype(np.float32)
        original_device = data.device
    else:
        data_np = data.astype(np.float32)

    # --- Original Logic (using data_np) ---
    x_shape = data_np.shape[0] // grid_density_factor
    y_shape = data_np.shape[1] // grid_density_factor
    z_shape = data_np.shape[2] // grid_density_factor

    # gridx = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    # gridy = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    # gridz = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor

    gridx = np.zeros((x_shape, y_shape, z_shape), dtype=np.float32)
    gridy = np.zeros((x_shape, y_shape, z_shape), dtype=np.float32)
    gridz = np.zeros((x_shape, y_shape, z_shape), dtype=np.float32)

    if DEVICE == 'cuda':
        transform = gryds.BSplineTransformationCuda([gridx, gridy, gridz])
        interpolator = gryds.BSplineInterpolatorCuda(data_np, order=3, mode="mirror")
    else:
        transform = gryds.BSplineTransformation([gridx, gridy, gridz])
        interpolator = gryds.Interpolator(data_np, order=3, mode="mirror")

    nifti_roi = interpolator.transform(transform)

    # --- Bridge: Convert back to Tensor if needed ---
    if is_tensor:
        return torch.from_numpy(nifti_roi).to(original_device)

    return nifti_roi


def get_deformed_sphere_fast(current_volume_tensor, intensity, pos, radius, height, margin,
                             grid_density_factor, deformation_factor):
    """
    Generates a deformed cylinder (Updated to support elongated height).
    """
    data_shape = current_volume_tensor.shape

    # 1. Update bounding box sizes to accommodate a long cylinder
    size_xy = 2 * radius + margin
    size_z = height + margin

    center_xy = size_xy // 2
    center_z = size_z // 2

    # # --- 1. Generate Small Cylinder (CPU / Numpy) ---
    # # Create a rectangular box instead of a perfect cube
    # small_cylinder_volume = np.zeros((size_xy, size_xy, size_z), dtype=np.float32)
    # cylinder_mask_np = create_cylinder(small_cylinder_volume, [center_xy, center_xy, center_z],
    #                                           radius, height)
    #
    # # Apply intensity
    # small_cylinder_volume[cylinder_mask_np > 0] = intensity

    # --- 1. Directly Generate the Torus Segment (CPU / Numpy) ---
    small_cylinder_volume = np.zeros((size_xy, size_xy, size_z), dtype=np.float32)

    # Call the torus function.
    # Example for ET Tube: radius=8 (outer), inner_radius=5 (hollow core)
    cylinder_mask_np = create_torus_segment(
        data=small_cylinder_volume,
        center=[center_xy, center_xy, center_z],
        tube_radius=radius,  # Outer wall radius
        height=height,
        curve_radius=400,
        inner_radius=radius - 3.0  # Optional: makes the tube hollow with a 3-voxel thick wall
    )

    # Apply intensity only to the plastic wall
    small_cylinder_volume[cylinder_mask_np > 0] = intensity

    # --- 2. Deform Small Cylinder (CPU) ---
    deformed_small_np = bspline(small_cylinder_volume, grid_density_factor, deformation_factor)

    # --- 3. Move Result to GPU ---
    deformed_small_cylinder = torch.from_numpy(deformed_small_np).float().to(DEVICE)

    # --- 4. Initialize Large Volume (PyTorch/GPU) ---
    cylinder_vol = torch.zeros_like(current_volume_tensor)

    x, y, z = pos

    # --- 5. Calculate raw coordinates (Using independent sizes) ---
    x_start = int(x - center_xy)
    x_end = int(x_start + size_xy)
    y_start = int(y - center_xy)
    y_end = int(y_start + size_xy)
    # z_start = int(z - center_z)
    # z_end = int(z_start + size_z)
    z_start = int(z)
    z_end = int(z_start + size_z)

    # --- 6. Clamp coordinates (Destination) ---
    sphere_x_start = max(0, x_start)
    sphere_x_end = min(data_shape[0], x_end)
    sphere_y_start = max(0, y_start)
    sphere_y_end = min(data_shape[1], y_end)
    sphere_z_start = max(0, z_start)
    sphere_z_end = min(data_shape[2], z_end)

    # --- 7. Calculate Source coordinates ---
    small_x_start = sphere_x_start - x_start
    small_x_end = small_x_start + (sphere_x_end - sphere_x_start)
    small_y_start = sphere_y_start - y_start
    small_y_end = small_y_start + (sphere_y_end - sphere_y_start)
    small_z_start = sphere_z_start - z_start
    small_z_end = small_z_start + (sphere_z_end - sphere_z_start)

    # --- 8. Apply Safe Assignment ---
    if (sphere_x_end > sphere_x_start and
            sphere_y_end > sphere_y_start and
            sphere_z_end > sphere_z_start):
        cylinder_vol[sphere_x_start:sphere_x_end,
        sphere_y_start:sphere_y_end,
        sphere_z_start:sphere_z_end] = deformed_small_cylinder[
                                       small_x_start:small_x_end,
                                       small_y_start:small_y_end,
                                       small_z_start:small_z_end]

    # --- 9. Create Mask ---
    mask = torch.round(cylinder_vol) != 0

    return cylinder_vol, mask


# def get_deformed_sphere_fast(current_volume_tensor, intensity, pos, radius, height, margin,
#                              grid_density_factor, deformation_factor):
#     """
#     Generates a deformed curved cylinder (ET Tube) safely.
#     Handles dynamic bounding box sizing and correct anatomical placement.
#     """
#     data_shape = current_volume_tensor.shape
#
#     # --- Fix 1: Realistic curve radius for anatomical correctness ---
#     # A larger radius creates a subtle, realistic bend instead of a sharp angle
#     curve_radius = 600.0
#
#     # --- Fix 2: Dynamic bounding box calculation ---
#     # Ensures the curved tube does not get clipped by the local Numpy array boundaries
#     safe_radius = max(curve_radius, height + 10.0)
#     max_shift = safe_radius - np.sqrt(safe_radius ** 2 - height ** 2)
#
#     size_xy = int(2 * radius + margin + (2 * max_shift) + 20)
#     size_z = height + margin
#
#     center_xy = size_xy // 2
#
#     # --- Fix 3: Start the tube near the top of the bounding box ---
#     # Gives the tube the full local Z-height to extend downwards
#     top_z = size_z - (margin // 2)
#
#     # Initialize local CPU volume
#     small_cylinder_volume = np.zeros((size_xy, size_xy, size_z), dtype=np.float32)
#
#     # Generate the curved tube segment
#     cylinder_mask_np = create_torus_segment(
#         data=small_cylinder_volume,
#         center=[center_xy, center_xy, top_z],
#         tube_radius=radius,
#         height=height,
#         curve_radius=safe_radius,
#         inner_radius=radius - 3.0
#     )
#
#     # Apply intensity to the solid mask
#     small_cylinder_volume[cylinder_mask_np > 0] = intensity
#
#     # --- Deform Small Cylinder (CPU) ---
#     deformed_small_np = bspline(small_cylinder_volume, grid_density_factor, deformation_factor)
#
#     # --- Move Result to GPU ---
#     deformed_small_cylinder = torch.from_numpy(deformed_small_np).float().to(DEVICE)
#
#     # --- Initialize Large Volume (PyTorch/GPU) ---
#     cylinder_vol = torch.zeros_like(current_volume_tensor)
#
#     x, y, z = pos
#
#     # --- Calculate raw destination coordinates ---
#     x_start = int(x - center_xy)
#     x_end = int(x_start + size_xy)
#     y_start = int(y - center_xy)
#     y_end = int(y_start + size_xy)
#
#     # --- Fix 4: Direction of placement ---
#     # The tube is placed extending DOWNWARDS from the sampled (z) position
#     z_start = int(z - size_z)
#     z_end = int(z)
#
#     # --- Clamp coordinates (Destination boundaries) ---
#     sphere_x_start = max(0, x_start)
#     sphere_x_end = min(data_shape[0], x_end)
#     sphere_y_start = max(0, y_start)
#     sphere_y_end = min(data_shape[1], y_end)
#     sphere_z_start = max(0, z_start)
#     sphere_z_end = min(data_shape[2], z_end)
#
#     # --- Calculate Source coordinates ---
#     small_x_start = sphere_x_start - x_start
#     small_x_end = small_x_start + (sphere_x_end - sphere_x_start)
#     small_y_start = sphere_y_start - y_start
#     small_y_end = small_y_start + (sphere_y_end - sphere_y_start)
#     small_z_start = sphere_z_start - z_start
#     small_z_end = small_z_start + (sphere_z_end - sphere_z_start)
#
#     # --- Apply Safe Assignment ---
#     if (sphere_x_end > sphere_x_start and
#             sphere_y_end > sphere_y_start and
#             sphere_z_end > sphere_z_start):
#         cylinder_vol[sphere_x_start:sphere_x_end,
#         sphere_y_start:sphere_y_end,
#         sphere_z_start:sphere_z_end] = deformed_small_cylinder[
#                                        small_x_start:small_x_end,
#                                        small_y_start:small_y_end,
#                                        small_z_start:small_z_end]
#
#     # --- Create Mask ---
#     mask = torch.round(cylinder_vol) != 0
#
#     return cylinder_vol, mask
