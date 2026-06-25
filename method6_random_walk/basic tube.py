import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import os
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt as edt_gpu
from scipy.ndimage import distance_transform_edt as edt_cpu
from scipy.ndimage import gaussian_filter1d

from seg import generate_full_airway_mask
import config as cfg
import drr

DEVICE = cfg.DEVICE

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
INPUT_CT_PATH = "../ct/ct_file1.nii.gz"
SEGS_DIR = "./segs"

OUTPUT_PATH_NII = "./output/curved_path_solid.nii.gz"
OUTPUT_TUBE_NII = "./output/curved_tube_hollow.nii.gz"
OUTPUT_PATH_DRR = "./output/drr_curved_path_solid.png"
OUTPUT_TUBE_DRR = "./output/drr_curved_tube_hollow.png"
OUTPUT_COMBINED_DRR = "./output/drr_ct_with_tube.png"

# ==========================================
# HYPERPARAMETERS
# ==========================================
TUBE_PLACEMENT = "LEFT"  # "LEFT" or "RIGHT"
TUBE_HEIGHT_FRACTION = 0.9
BLOCK_SIZE = 30

TUBE_DIAMETER = 8.0
TUBE_THICKNESS = 2.0
TUBE_INTENSITY = 1000.0

PATH_SMOOTHING_SIGMA = 9.0
WALK_STEP_STD = 4.0  # Std (voxels) of each random X/Y step in the safe walk

# Biased Coin Parameters
DRIFT_STRENGTH_MIN = 0.55  # Minimum lateral pull
DRIFT_STRENGTH_MAX = 7.0  # Maximum lateral pull

# Shaving Parameters
SHAVING_BOOL = False
SHAVING_SIGMA = 1.1


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


def drr_save(volume, path, multiplier=1.0):
    """Run DRR projection and save to disk."""
    t = torch.from_numpy(volume * multiplier).float().to(DEVICE)
    drr.save_drr(drr.apply_drr_post_processing(drr.create_drr_from_ct(t)), path)


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

    top_z_lo = max(z_min, z_max - block_size + 1)
    top_pt = random_seg_point(mask, top_z_lo, z_max)

    z_target = max(z_min, int(z_max - (z_max - z_min) * height_fraction))

    return top_pt, z_target


# ==========================================
# CENTERLINE — BIASED RANDOM WALK
# ==========================================
def extract_centerline(mask, pt_top, z_target, placement, sigma, tube_radius, walk_step_std):
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

    x0, y0, z0 = pt_top
    z_hi = int(z0)
    z_lo = int(z_target)

    MAX_TRIES = 20
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

    # --- Walk slice-by-slice ---
    for z in range(z_hi, z_lo - 1, -1):
        accepted = False

        for _ in range(MAX_TRIES):
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
            pass  # cx, cy unchanged

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
def gaussian_smooth_3d_gpu(vol, sigma):
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
def build_tube_gpu(shape, px, py, pz, diameter, thickness, shaving_bool, shave_sigma):
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
    eroded = (F.conv2d(slices, disk_kernel, padding=r) >= disk_area).float()
    shell = ((slices > 0) & (eroded < 1)).float()

    hollow = shell.squeeze(1).permute(1, 2, 0) * TUBE_INTENSITY

    return solid.cpu().numpy().astype(np.uint8), hollow.cpu().numpy().astype(np.float32)


# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs("./output", exist_ok=True)

    ct_img = nib.load(INPUT_CT_PATH)
    ct_data = ct_img.get_fdata()
    mask = nib.load(generate_full_airway_mask(INPUT_CT_PATH, SEGS_DIR)).get_fdata().astype(np.uint8)

    print(f"Sampling endpoints (block={BLOCK_SIZE}, placement={TUBE_PLACEMENT})...")
    top, z_target = get_start_point_and_depth(mask, TUBE_HEIGHT_FRACTION, BLOCK_SIZE)
    print(f"  Top:    {top}")
    print(f"  Target Depth (Z): {z_target} ({TUBE_PLACEMENT})")

    print("Extracting centerline (biased random walk + EDT)...")
    px, py, pz = extract_centerline(
        mask, top, z_target, TUBE_PLACEMENT,
        sigma=PATH_SMOOTHING_SIGMA,
        tube_radius=TUBE_DIAMETER / 2.0,
        walk_step_std=WALK_STEP_STD,
    )

    print(f"Building tube on {DEVICE}...")
    path_solid, tube_hollow = build_tube_gpu(
        ct_img.shape, px, py, pz, TUBE_DIAMETER, TUBE_THICKNESS, SHAVING_BOOL, SHAVING_SIGMA,
    )

    tube_hollow[mask == 0] = 0

    print("Saving NIfTI volumes...")
    aff, hdr = ct_img.affine, ct_img.header
    nib.save(nib.Nifti1Image(path_solid, aff, hdr), OUTPUT_PATH_NII)
    nib.save(nib.Nifti1Image(tube_hollow.astype(np.int16), aff, hdr), OUTPUT_TUBE_NII)

    print("Generating DRR projections...")
    drr_save(path_solid.astype(np.float32), OUTPUT_PATH_DRR, multiplier=1000.0)
    drr_save(tube_hollow, OUTPUT_TUBE_DRR)

    combined = np.copy(ct_data)
    combined[tube_hollow > 0] = TUBE_INTENSITY
    drr_save(combined, OUTPUT_COMBINED_DRR)

    print("Done.")


if __name__ == "__main__":
    main()