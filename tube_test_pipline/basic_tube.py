import numpy as np
import torch
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt as edt_gpu
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List

import config as cfg


# =============================================================================
# ENDPOINT SELECTION
# =============================================================================
def get_random_voxel_in_z_range(mask: np.ndarray, z_min: int, z_max: int):
    """Pick a random voxel from the segmentation mask within the given Z range."""
    z_block = mask[:, :, z_min:z_max + 1]
    x_indices, y_indices, z_indices_local = np.where(z_block > 0)

    if len(x_indices) == 0:
        raise ValueError(f"No segmentation voxels found in Z range [{z_min}, {z_max}]!")

    random_idx = np.random.randint(len(x_indices))
    return (
        int(x_indices[random_idx]),
        int(y_indices[random_idx]),
        int(z_indices_local[random_idx]) + z_min
    )


def _mask_hemisphere(volume_block: np.ndarray, keep_hemisphere: str) -> np.ndarray:
    """Zeroes out one half of the volume block along the X axis based on 'LEFT' or 'RIGHT'."""
    block_copy = volume_block.copy()
    segmentation_coords = np.argwhere(block_copy > 0)

    if len(segmentation_coords) > 0:
        x_midpoint = int(np.mean(segmentation_coords[:, 0]))
        if keep_hemisphere == "LEFT":
            block_copy[x_midpoint:, :, :] = 0
        elif keep_hemisphere == "RIGHT":
            block_copy[:x_midpoint, :, :] = 0

    return block_copy


def get_top_endpoint(mask: np.ndarray, z_min: int, z_max: int, block_size: int):
    """Finds a random starting point in the uppermost `block_size` slices."""
    top_z_min = max(z_min, z_max - block_size + 1)
    return get_random_voxel_in_z_range(mask, top_z_min, z_max)


def get_bottom_endpoint(mask: np.ndarray, z_min: int, z_max: int, height_fraction: float, block_size: int,
                        placement: str):
    """Finds a target point deep in the mask, optionally biased to LEFT or RIGHT."""
    z_target = max(z_min, int(z_max - (z_max - z_min) * height_fraction))
    bot_z_min = max(0, z_target - block_size // 2)
    bot_z_max = min(mask.shape[2] - 1, z_target + block_size // 2)

    if placement in ("LEFT", "RIGHT"):
        bottom_volume_block = mask[:, :, bot_z_min:bot_z_max + 1]
        masked_block = _mask_hemisphere(bottom_volume_block, placement)

        if not np.any(masked_block > 0):
            print(f"  Warning: {placement} branch empty in target block, falling back to full block (MAIN).")
            return get_random_voxel_in_z_range(mask, bot_z_min, bot_z_max)
        else:
            x_indices, y_indices, z_indices_local = np.where(masked_block > 0)
            random_idx = np.random.randint(len(x_indices))
            return (
                int(x_indices[random_idx]),
                int(y_indices[random_idx]),
                int(z_indices_local[random_idx]) + bot_z_min
            )

    # "MAIN" or fallback
    return get_random_voxel_in_z_range(mask, bot_z_min, bot_z_max)


def find_endpoints(mask: np.ndarray, height_fraction: float, placement: str, block_size: int):
    """
    Returns (top_point, bottom_point, actual_placement).
    - top_point: random point from the top `block_size` slices (no L/R filtering).
    - bottom_point: random point from the bottom block, optionally filtered to LEFT/RIGHT.
    """
    coords = np.argwhere(mask > 0)
    z_max, z_min = int(coords[:, 2].max()), int(coords[:, 2].min())

    top_point = get_top_endpoint(mask, z_min, z_max, block_size)

    if placement == "RANDOM":
        placement = np.random.choice(["LEFT", "RIGHT"])
        print(f"  Randomized branch selected: {placement}")

    bottom_point = get_bottom_endpoint(mask, z_min, z_max, height_fraction, block_size, placement)

    return top_point, bottom_point, placement


# =============================================================================
# CENTERLINE EXTRACTION
# =============================================================================
def compute_edt_map(mask: np.ndarray) -> np.ndarray:
    """Computes the Euclidean Distance Transform (EDT) on GPU."""
    torch.cuda.empty_cache()  # free PyTorch's cached blocks for CuPy
    mask_gpu = cp.asarray(mask, dtype=cp.uint8)
    edt_map = cp.asnumpy(edt_gpu(mask_gpu))
    del mask_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return edt_map


def trace_path_through_edt(edt_map: np.ndarray, top_point: Tuple[int, int, int],
                           bottom_point: Tuple[int, int, int], search_radius: int) -> Tuple[
    List[float], List[float], List[int]]:
    """Walks slice-by-slice from top to bottom, snapping to the local maxima of the EDT map."""
    top_x, top_y, top_z = top_point
    bot_x, bot_y, bot_z = bottom_point

    path_x, path_y, path_z = [], [], []
    z_max = int(max(top_z, bot_z))
    z_min = int(min(top_z, bot_z))
    z_distance_total = abs(top_z - bot_z) + 1e-6

    for current_z in range(z_max, z_min - 1, -1):
        # Linear interpolation
        t = abs(current_z - top_z) / z_distance_total
        interpolated_x = top_x + t * (bot_x - top_x)
        interpolated_y = top_y + t * (bot_y - top_y)

        slice_edt_map = edt_map[:, :, current_z]

        # Define search window
        x_start = max(0, int(interpolated_x - search_radius))
        x_end = min(slice_edt_map.shape[0], int(interpolated_x + search_radius + 1))
        y_start = max(0, int(interpolated_y - search_radius))
        y_end = min(slice_edt_map.shape[1], int(interpolated_y + search_radius + 1))

        local_search_window = slice_edt_map[x_start:x_end, y_start:y_end]

        if np.max(local_search_window) > 0:
            max_intensity_index = np.unravel_index(np.argmax(local_search_window), local_search_window.shape)
            path_x.append(x_start + max_intensity_index[0])
            path_y.append(y_start + max_intensity_index[1])
        else:
            # Fallback to interpolated coordinates if no valid EDT values in window
            path_x.append(interpolated_x)
            path_y.append(interpolated_y)

        path_z.append(current_z)

    return path_x, path_y, path_z


def smooth_path(path_x: List[float], path_y: List[float], path_z: List[int], sigma: float) -> Tuple[
    np.ndarray, np.ndarray, List[int]]:
    """Applies Gaussian smoothing to the X and Y coordinates of the path."""
    smoothed_x = gaussian_filter1d(path_x, sigma=sigma)
    smoothed_y = gaussian_filter1d(path_y, sigma=sigma)
    return smoothed_x, smoothed_y, path_z


def extract_centerline(mask: np.ndarray, top_point: Tuple[int, int, int], bottom_point: Tuple[int, int, int],
                       search_radius: int, smoothing_sigma: float):
    """
    Extracts a smooth, safe centerline through the mask by interpolating between endpoints
    and snapping to the thickest regions (EDT).
    """
    edt_map = compute_edt_map(mask)
    path_x, path_y, path_z = trace_path_through_edt(edt_map, top_point, bottom_point, search_radius)
    return smooth_path(path_x, path_y, path_z, smoothing_sigma)


# =============================================================================
# TUBE GEOMETRY GENERATION
# =============================================================================
def create_1d_gaussian_kernel(sigma: float, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Creates a 1D Gaussian kernel for smoothing."""
    kernel_size = int(6 * sigma + 1) | 1  # Force odd
    half_kernel = kernel_size // 2
    kernel_coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - half_kernel
    gaussian_weights = torch.exp(-0.5 * (kernel_coords / sigma) ** 2)
    gaussian_weights = gaussian_weights / gaussian_weights.sum()
    return gaussian_weights, half_kernel


def apply_separable_conv3d(volume_tensor: torch.Tensor, gaussian_weights: torch.Tensor,
                           half_kernel: int) -> torch.Tensor:
    """Applies a separable 3D convolution to the volume."""
    v = volume_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)

    # F.pad order for 5D: (Z_left, Z_right, Y_left, Y_right, X_left, X_right)
    v = F.conv3d(F.pad(v, (0, 0, 0, 0, half_kernel, half_kernel), mode='replicate'),
                 gaussian_weights.view(1, 1, -1, 1, 1))
    v = F.conv3d(F.pad(v, (0, 0, half_kernel, half_kernel, 0, 0), mode='replicate'),
                 gaussian_weights.view(1, 1, 1, -1, 1))
    v = F.conv3d(F.pad(v, (half_kernel, half_kernel, 0, 0, 0, 0), mode='replicate'),
                 gaussian_weights.view(1, 1, 1, 1, -1))

    return v.squeeze(0).squeeze(0)


def gaussian_smooth_3d_gpu(volume_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur via three 1D convolutions on GPU."""
    gaussian_weights, half_kernel = create_1d_gaussian_kernel(sigma, volume_tensor.device)
    return apply_separable_conv3d(volume_tensor, gaussian_weights, half_kernel)


def create_solid_path(shape: Tuple[int, int, int], path_x: List[float], path_y: List[float],
                      path_z: List[int], diameter: float, device: torch.device) -> torch.Tensor:
    """Stamps circles along the path to create a solid tube volume."""
    radius_sq = (diameter / 2.0) ** 2

    # XY coordinate grids
    x_grid = torch.arange(shape[0], device=device, dtype=torch.float32).view(-1, 1)
    y_grid = torch.arange(shape[1], device=device, dtype=torch.float32).view(1, -1)

    solid_volume = torch.zeros(shape, device=device, dtype=torch.float32)
    for i, z in enumerate(path_z):
        dist_sq = (x_grid - path_x[i]) ** 2 + (y_grid - path_y[i]) ** 2
        solid_volume[:, :, z][dist_sq <= radius_sq] = 1.0

    return solid_volume


def hollow_out_solid(solid_volume: torch.Tensor, thickness: float, tube_intensity: float,
                     device: torch.device) -> torch.Tensor:
    """Hollows out the solid volume using batched 2D morphological erosion."""
    radius = int(np.ceil(thickness))
    diameter = 2 * radius + 1

    cy, cx = torch.meshgrid(
        torch.arange(diameter, device=device) - radius,
        torch.arange(diameter, device=device) - radius, indexing='ij')

    disk = (cx.float() ** 2 + cy.float() ** 2 <= thickness ** 2).float()
    disk_kernel = disk.unsqueeze(0).unsqueeze(0)  # (1, 1, d, d)
    disk_area = disk.sum()

    # Treat Z as batch dim -> one conv2d call for ALL slices
    slices = solid_volume.permute(2, 0, 1).unsqueeze(1)  # (Z, 1, X, Y)
    eroded = (F.conv2d(slices, disk_kernel, padding=radius) >= disk_area).float()

    shell = ((slices > 0) & (eroded < 1)).float()  # thin wall only
    hollow_volume = shell.squeeze(1).permute(1, 2, 0) * tube_intensity  # (X, Y, Z)

    return hollow_volume


def build_tube_gpu(shape: Tuple[int, int, int], path_x: List[float], path_y: List[float], path_z: List[int],
                   diameter: float, thickness: float, shave_sigma: float, tube_intensity: float,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Builds solid and hollow tube entirely on GPU, returning them as numpy arrays."""

    # 1. Stamp circles along the path
    solid_volume = create_solid_path(shape, path_x, path_y, path_z, diameter, device)

    # 2. Gaussian smooth -> threshold (removes voxel staircase)
    solid_volume = (gaussian_smooth_3d_gpu(solid_volume, shave_sigma) >= 0.5).float()

    # 3. Hollow via batched 2D morphological erosion
    hollow_volume = hollow_out_solid(solid_volume, thickness, tube_intensity, device)

    return solid_volume.cpu().numpy().astype(np.uint8), hollow_volume.cpu().numpy().astype(np.float32)


# =============================================================================
# EXPOSED API
# =============================================================================
def build_tube_volume(ct_shape: Tuple[int, int, int], mask_data: np.ndarray, params: dict) -> np.ndarray:
    """
    Builds the 3D tube volume (hollow) based on the input parameters and segmentation mask.
    Returns: hollow_volume (numpy array).
    """
    device = cfg.DEVICE

    # Extract params
    placement = params["TUBE_PLACEMENT"]
    height_fraction = params["TUBE_HEIGHT_FRACTION"]
    block_size = params["BLOCK_SIZE"]
    diameter = params["TUBE_DIAMETER"]
    thickness = params["TUBE_THICKNESS"]
    intensity = params["TUBE_INTENSITY"]
    search_radius = params["PATH_SEARCH_RADIUS"]

    print(f"Sampling endpoints (block={block_size}, placement={placement})...")
    top_point, bottom_point, actual_placement = find_endpoints(
        mask_data, height_fraction, placement, block_size
    )
    print(f"  Top:    {top_point}")
    print(f"  Bottom: {bottom_point} ({actual_placement})")

    print("Extracting safe centerline (GPU EDT)...")
    path_x, path_y, path_z = extract_centerline(
        mask_data, top_point, bottom_point, search_radius, cfg.PATH_SMOOTHING_SIGMA
    )

    z_top_ct = ct_shape[2] - 1
    current_top_z = path_z[0]
    current_top_x = path_x[0]
    current_top_y = path_y[0]

    if current_top_z < z_top_ct:
        print(f"Extending tube straight up from Z={current_top_z} to Z={z_top_ct}...")
        # Create coordinates for the straight line going up
        ext_z = list(range(z_top_ct, current_top_z, -1))
        ext_x = np.full(len(ext_z), current_top_x)
        ext_y = np.full(len(ext_z), current_top_y)

        # Prepend the extension to the original smoothed paths
        path_x = np.concatenate((ext_x, path_x))
        path_y = np.concatenate((ext_y, path_y))
        path_z = ext_z + path_z

    print(f"Building tube on {device}...")
    solid_volume, hollow_volume = build_tube_gpu(
        ct_shape, path_x, path_y, path_z,
        diameter, thickness, cfg.SHAVING_SIGMA, intensity, device
    )

    print("Sanity check: clipping tube to segmentation...")

    # hollow_volume[mask_data == 0] = 0

    clip_margin = 20  # Number of slices to skip clipping near the top
    clip_limit_z = max(0, current_top_z - clip_margin)
    for z in range(clip_limit_z):
        slice_mask = mask_data[:, :, z]
        hollow_volume[:, :, z][slice_mask == 0] = 0

    return hollow_volume
