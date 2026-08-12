"""
Intubation tube generation and insertion into CT volumes.
Extracted from tube_test6/basic tube.py.

All tube hyperparameters are kept here (not in config.py) for quick iteration.
Main entry point: add_tube_to_ct()
"""

import numpy as np
import torch
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt as edt_gpu
from scipy.ndimage import distance_transform_edt as edt_cpu
from scipy.ndimage import gaussian_filter1d
import math
import sys
import os

# Add the project root to sys.path to resolve PyCharm import errors
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import config as cfg

DEVICE = cfg.DEVICE

# ==========================================
# TUBE HYPERPARAMETERS
# ==========================================
BLOCK_SIZE = 20

# Biased Coin Parameters
DRIFT_STRENGTH_MIN = 0.55  # Minimum lateral pull
DRIFT_STRENGTH_MAX = 7.0  # Maximum lateral pull

# Shaving Parameters
SHAVING_BOOL = False
SHAVING_SIGMA = 1.1

# Random Walk Retry Parameters
MAX_CIRCLE_TRIES = 5000  # Max attempts to find a safe circle at each z-slice
MAX_CONSECUTIVE_FAILURES = 50  # Number of consecutive z-slice failures before giving up on this walk
TUBE_RESTART_ATTEMPTS = 5  # How many times to restart with a smaller diameter before failing entirely
TUBE_DIAMETER_SHRINK_STEP = 1.0  # How much to reduce the diameter on each restart


class TubeWalkFailure(Exception):
    """Raised when the random walk cannot find safe circle placements."""
    pass


# ==========================================
# HELPERS
# ==========================================
def random_seg_point(mask, z_lo, z_hi):
    """Pick a random voxel from mask where z is in [z_lo, z_hi]."""
    block = mask[:, :, z_lo:z_hi + 1]
    xs, ys, zs = np.where(block > 0)
    if len(xs) == 0:
        raise ValueError(f"No segmentation voxels in z=[{z_lo}, {z_hi}]!")
    i = np.random.randint(len(xs))
    return int(xs[i]), int(ys[i]), int(zs[i]) + z_lo


# ==========================================
# FIND STARTING POINT & DEPTH
# ==========================================
def get_start_point_and_depth(mask, height_fraction, block_size):
    """
    Returns (top_pt, z_target).
    The bottom point is no longer fixed; the walk will drift naturally to the requested side.
    """
    coords = np.argwhere(mask > 0)
    z_max, z_min = int(coords[:, 2].max()), int(coords[:, 2].min())

    # Add a margin of 10 voxels to abandon the very top part
    effective_z_max = max(z_min, z_max - 10)

    top_z_lo = max(z_min, effective_z_max - block_size + 1)
    top_pt = random_seg_point(mask, top_z_lo, effective_z_max)

    z_target = max(z_min, int(z_max - (z_max - z_min) * height_fraction))

    return top_pt, z_target


# ==========================================
# CENTERLINE — BIASED RANDOM WALK
# ==========================================
def extract_centerline(mask, pt_top, z_target, placement, sigma, tube_radius, walk_step_std,
                       max_circle_tries=None, max_consecutive_failures=None):
    """
    Traces a centerline from the top endpoint down to z_target.
    Applies a constant biased "coin flip" to the X-axis to pull the walker left or right.
    """
    if placement not in ("LEFT", "RIGHT"):
        raise ValueError(f"Invalid TUBE_PLACEMENT '{placement}'. Must be 'LEFT' or 'RIGHT'.")

    # --- EDT (GPU preferred, CPU fallback) ---
    try:
        torch.cuda.empty_cache()
        mask_gpu = cp.asarray(mask, dtype=cp.uint8)
        edt = cp.asnumpy(edt_gpu(mask_gpu))
        del mask_gpu
        cp.get_default_memory_pool().free_all_blocks()
        print("  (EDT ran on GPU)")
    except cp.cuda.memory.OutOfMemoryError:
        print("  GPU OOM for EDT — falling back to CPU...")
        edt = edt_cpu(mask)

    if max_circle_tries is None:
        max_circle_tries = MAX_CIRCLE_TRIES
    if max_consecutive_failures is None:
        max_consecutive_failures = MAX_CONSECUTIVE_FAILURES

    x0, y0, z0 = pt_top
    z_hi = int(z0)
    z_lo = int(z_target)

    path_x, path_y, path_z = [], [], []

    # --- Set Up the Biased Coin ---
    # Randomize the bias strength to ensure dataset variance
    bias_magnitude = np.random.uniform(DRIFT_STRENGTH_MIN, DRIFT_STRENGTH_MAX)

    # Apply the direction (Assuming X splits Left/Right)
    # Switch 1.0 and -1.0 if the anatomical directions in your NIfTI are reversed
    x_drift = bias_magnitude if placement == "RIGHT" else -bias_magnitude
    y_drift = 0.0  # Keep Y unguided (pure random) or add small drift if anterior/posterior bias is needed

    print(f"  Walk Drift: X-axis biased by {x_drift:.2f} voxels/step ({placement})")

    # --- Initialise walker ---
    cx, cy = float(x0), float(y0)

    if edt[int(round(cx)), int(round(cy)), z_hi] < tube_radius:
        mi = np.unravel_index(np.argmax(edt[:, :, z_hi]), edt[:, :, z_hi].shape)
        cx, cy = float(mi[0]), float(mi[1])
        print(f"  Start snapped to EDT-max: ({cx:.0f}, {cy:.0f}) at z={z_hi}")

    # --- Walk slice-by-slice UPWARDS (straight line + noise, no EDT check) ---
    z_vol_max = mask.shape[2] - 1
    up_path_x, up_path_y, up_path_z = [], [], []
    up_cx, up_cy = cx, cy
    for z in range(z_hi + 1, z_vol_max + 1):
        # Straight line upwards with independent normal noise (becomes a gentle wiggle after smoothing)
        wiggle_x = np.random.normal(0, walk_step_std)
        wiggle_y = np.random.normal(0, walk_step_std)
        up_path_x.append(up_cx + wiggle_x)
        up_path_y.append(up_cy + wiggle_y)
        up_path_z.append(z)

    # Reverse upward path so it connects seamlessly to the downward path
    up_path_x.reverse()
    up_path_y.reverse()
    up_path_z.reverse()

    path_x.extend(up_path_x)
    path_y.extend(up_path_y)
    path_z.extend(up_path_z)

    # --- Walk slice-by-slice DOWNWARDS ---
    consecutive_failures = 0
    for z in range(z_hi, z_lo - 1, -1):
        accepted = False

        for _ in range(max_circle_tries):
            # The Biased Coin: Shift the mean of the normal distribution
            dx = np.random.normal(x_drift, walk_step_std)
            dy = np.random.normal(y_drift, walk_step_std)
            nx, ny = cx + dx, cy + dy
            ix, iy = int(round(nx)), int(round(ny))

            if (0 <= ix < edt.shape[0] and
                    0 <= iy < edt.shape[1] and
                    edt[ix, iy, z] >= tube_radius):
                cx, cy = nx, ny
                accepted = True
                break

        if not accepted:
            consecutive_failures += 1
            print(
                f"  WARNING: No safe placement at z={z} (consecutive failures: {consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                raise TubeWalkFailure(
                    f"Random walk stuck: {consecutive_failures} consecutive z-slices "
                    f"with no safe circle (tube_radius={tube_radius:.1f}). "
                    f"Restart with smaller diameter needed."
                )
        else:
            consecutive_failures = 0

        path_x.append(cx)
        path_y.append(cy)
        path_z.append(z)

    print(f"  Centerline traced: {len(path_z)} points (walk_step_std={walk_step_std})")

    return (
        gaussian_filter1d(path_x, sigma=sigma),
        gaussian_filter1d(path_y, sigma=sigma),
        path_z,
    )


# ==========================================
# GPU: 3D Gaussian smoothing (separable)
# ==========================================
@torch.no_grad()
def gaussian_smooth_3d_gpu(vol, sigma):
    """Functionality for gaussian_smooth_3d_gpu."""
    ks = int(6 * sigma + 1) | 1
    half = ks // 2
    t = torch.arange(ks, device=vol.device, dtype=torch.float32) - half
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = g / g.sum()

    v = vol.unsqueeze(0).unsqueeze(0)

    v = F.conv3d(F.pad(v, (0, 0, 0, 0, half, half), mode='replicate'), g.view(1, 1, -1, 1, 1))
    v = F.conv3d(F.pad(v, (0, 0, half, half, 0, 0), mode='replicate'), g.view(1, 1, 1, -1, 1))
    v = F.conv3d(F.pad(v, (half, half, 0, 0, 0, 0), mode='replicate'), g.view(1, 1, 1, 1, -1))

    return v.squeeze(0).squeeze(0)


# ==========================================
# GPU: Build tube volumes
# ==========================================
@torch.no_grad()
def build_tube_gpu(shape, px, py, pz, diameter, thickness, shaving_bool, shave_sigma, tube_intensity):
    """Functionality for build_tube_gpu."""
    radius_sq = (diameter / 2.0) ** 2

    xg = torch.arange(shape[0], device=DEVICE, dtype=torch.float32).view(-1, 1)
    yg = torch.arange(shape[1], device=DEVICE, dtype=torch.float32).view(1, -1)

    solid = torch.zeros(shape, device=DEVICE, dtype=torch.float32)
    for i, z_val in enumerate(pz):
        z_idx = int(z_val)
        if z_idx < 0 or z_idx >= shape[2]:
            continue
        dist_sq = (xg - px[i]) ** 2 + (yg - py[i]) ** 2
        solid[:, :, z_idx][dist_sq <= radius_sq] = 1.0

    if shaving_bool:
        print(f"  Applying 3D shaving (sigma={shave_sigma})...")
        solid = (gaussian_smooth_3d_gpu(solid, shave_sigma) >= 0.5).float()

    r = int(np.ceil(thickness))
    d = 2 * r + 1
    cy_grid, cx_grid = torch.meshgrid(
        torch.arange(d, device=DEVICE) - r,
        torch.arange(d, device=DEVICE) - r,
        indexing='ij',
    )
    disk = (cx_grid.float() ** 2 + cy_grid.float() ** 2 <= thickness ** 2).float()
    disk_kernel = disk.unsqueeze(0).unsqueeze(0)
    disk_area = disk.sum()

    slices = solid.permute(2, 0, 1).unsqueeze(1)

    # Process convolution in chunks to save memory
    eroded = torch.zeros_like(slices)
    chunk_size = 32
    for start_idx in range(0, slices.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, slices.shape[0])
        eroded[start_idx:end_idx] = (
                F.conv2d(slices[start_idx:end_idx], disk_kernel, padding=r) >= disk_area).float()

    shell = ((slices > 0) & (eroded < 1)).float()

    hollow = shell.squeeze(1).permute(1, 2, 0) * tube_intensity

    return solid.cpu().numpy().astype(np.uint8), hollow.cpu().numpy().astype(np.float32)


# ==========================================
# MAIN ENTRY POINT: Add tube to CT volume
# ==========================================
@torch.no_grad()
def add_tube_to_ct(ct_data, trachea_mask, tube_diameter, tube_thickness):
    """
    Generates a random intubation tube and inserts it into the CT volume.
    If the random walk fails (no safe circle placements), the process restarts
    with a smaller tube diameter, up to TUBE_RESTART_ATTEMPTS times.

    Parameters
    ----------
    ct_data : torch.Tensor
        The 3D CT volume on GPU (will be modified in-place).
    trachea_mask : numpy.ndarray
        The trachea segmentation mask (uint8, on CPU).
    tube_diameter : float
        The diameter of the tube.
    tube_thickness : int
        The thickness of the tube walls.

    Returns
    -------
    ct_data : torch.Tensor
        The modified CT volume with the tube inserted.
    used_diameter : float
        The actual tube diameter used (may be smaller than requested if restarts occurred).
    used_thickness : int
        The actual tube thickness used.
    """
    from datagen.tube_randomization import get_random_tube_params

    mask_np = trachea_mask.astype(np.uint8) if not trachea_mask.dtype == np.uint8 else trachea_mask

    if not np.any(mask_np > 0):
        print("[TUBE] WARNING: Trachea mask is completely empty. Skipping tube insertion.")
        return ct_data, tube_diameter, tube_thickness

    current_diameter = tube_diameter
    current_thickness = tube_thickness

    for attempt in range(TUBE_RESTART_ATTEMPTS + 1):
        try:
            # Get random parameters for THIS specific tube instance
            params = get_random_tube_params(current_diameter)

            print(f"[TUBE] Attempt {attempt + 1}/{TUBE_RESTART_ATTEMPTS + 1} — "
                  f"Dia={current_diameter:.1f}, Thick={current_thickness}, "
                  f"Int={params['intensity']:.0f}, Sigma={params['path_smoothing_sigma']:.1f}")
            print(f"[TUBE] Sampling endpoints (block={BLOCK_SIZE}, placement={params['placement']})...")
            top, z_target = get_start_point_and_depth(mask_np, params['height_fraction'], BLOCK_SIZE)
            print(f"  Top:    {top}")
            print(f"  Target Depth (Z): {z_target} ({params['placement']})")

            print("[TUBE] Extracting centerline (biased random walk + EDT)...")
            px, py, pz = extract_centerline(
                mask_np, top, z_target, params['placement'],
                sigma=params['path_smoothing_sigma'],
                tube_radius=current_diameter / 2.0,
                walk_step_std=params['walk_step_std'],
            )

            ct_shape = ct_data.shape if isinstance(ct_data, torch.Tensor) else ct_data.shape
            print(f"[TUBE] Building tube on {DEVICE}...")
            path_solid, tube_hollow = build_tube_gpu(
                ct_shape, px, py, pz, current_diameter, current_thickness,
                SHAVING_BOOL, SHAVING_SIGMA,
                tube_intensity=params['intensity']
            )

            # Mask tube to trachea region only for the downward walk (z <= z_hi).
            # The upward extrapolation (z > z_hi) ignores the mask so it isn't cut short.
            z_hi = top[2]
            tube_hollow[:, :, :z_hi + 1][mask_np[:, :, :z_hi + 1] == 0] = 0

            # Insert tube into CT volume
            tube_mask = tube_hollow > 0
            if isinstance(ct_data, torch.Tensor):
                tube_tensor = torch.from_numpy(tube_hollow).float().to(ct_data.device)
                ct_data[tube_tensor > 0] = tube_tensor[tube_tensor > 0]
            else:
                ct_data[tube_mask] = tube_hollow[tube_mask]

            print("[TUBE] Tube inserted successfully.")
            return ct_data, current_diameter, current_thickness

        except TubeWalkFailure as e:
            print(f"[TUBE] Walk failed: {e}")

            if attempt < TUBE_RESTART_ATTEMPTS:
                current_diameter = max(2.0, current_diameter - TUBE_DIAMETER_SHRINK_STEP)
                current_thickness = max(1, math.ceil(current_diameter / 2))
                print(f"[TUBE] Restarting with smaller tube — "
                      f"new diameter={current_diameter:.1f}, new thickness={current_thickness}")
            else:
                print(f"[TUBE] WARNING: All {TUBE_RESTART_ATTEMPTS + 1} attempts exhausted. "
                      f"Skipping tube insertion for this volume.")
                return ct_data, current_diameter, current_thickness
