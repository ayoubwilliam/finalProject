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
TUBE_PLACEMENT = "RANDOM"       # "LEFT", "RIGHT", or "RANDOM"
TUBE_HEIGHT_FRACTION = 0.8
BLOCK_SIZE = 30                 # Z-slice thickness for top/bottom anchor blocks

TUBE_DIAMETER = 12.0
TUBE_THICKNESS = 3.0
TUBE_INTENSITY = 1000.0

PATH_SEARCH_RADIUS = 15
PATH_SMOOTHING_SIGMA = 20.0
SHAVING_SIGMA = 2.5


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
# FIND ENDPOINTS (CPU — trivial indexing)
# ==========================================
def find_endpoints(mask, height_fraction, placement, block_size):
    """
    Returns (top_pt, bot_pt, placement).
    - top_pt: random point from the top `block_size` slices (no L/R filtering).
    - bot_pt: random point from the bottom block, optionally filtered to LEFT/RIGHT.
    """
    coords = np.argwhere(mask > 0)
    z_max, z_min = int(coords[:, 2].max()), int(coords[:, 2].min())

    # --- Top: random point from uppermost block_size slices ---
    top_z_lo = max(z_min, z_max - block_size + 1)
    top_pt = random_seg_point(mask, top_z_lo, z_max)

    # --- Bottom: random point around the target depth ---
    z_target = max(z_min, int(z_max - (z_max - z_min) * height_fraction))
    bot_z_lo = max(0, z_target - block_size // 2)
    bot_z_hi = min(mask.shape[2] - 1, z_target + block_size // 2)

    # Resolve RANDOM to LEFT or RIGHT
    if placement == "RANDOM":
        placement = np.random.choice(["LEFT", "RIGHT"])
        print(f"  Randomized branch: {placement}")

    # For LEFT/RIGHT, mask out one half of the bottom block
    if placement in ("LEFT", "RIGHT"):
        bot_block = mask[:, :, bot_z_lo:bot_z_hi + 1].copy()
        seg_coords = np.argwhere(bot_block > 0)

        if len(seg_coords) > 0:
            x_mid = int(np.mean(seg_coords[:, 0]))
            if placement == "LEFT":
                bot_block[x_mid:, :, :] = 0
            else:
                bot_block[:x_mid, :, :] = 0

        # Fallback if nothing left after masking
        if not np.any(bot_block > 0):
            print(f"  Warning: {placement} branch empty, falling back to MAIN.")
            bot_pt = random_seg_point(mask, bot_z_lo, bot_z_hi)
        else:
            xs, ys, zs = np.where(bot_block > 0)
            i = np.random.randint(len(xs))
            bot_pt = int(xs[i]), int(ys[i]), int(zs[i]) + bot_z_lo
    else:
        bot_pt = random_seg_point(mask, bot_z_lo, bot_z_hi)

    return top_pt, bot_pt, placement


# ==========================================
# CENTERLINE (EDT on GPU via CuPy, auto-fallback to CPU)
# ==========================================
def extract_centerline(mask, pt_top, pt_bot, search_radius, sigma):
    """Walk slice-by-slice from top to bottom, snapping to the safest center."""
    try:
        torch.cuda.empty_cache()                # free PyTorch's cached blocks for CuPy
        mask_gpu = cp.asarray(mask, dtype=cp.uint8)  # bool mask → 1 byte/voxel (not 8)
        edt = cp.asnumpy(edt_gpu(mask_gpu))
        del mask_gpu; cp.get_default_memory_pool().free_all_blocks()
        print("  (EDT ran on GPU)")
    except cp.cuda.memory.OutOfMemoryError:
        print("  GPU OOM for EDT — falling back to CPU...")
        edt = edt_cpu(mask)
    x0, y0, z0 = pt_top
    x1, y1, z1 = pt_bot

    path_x, path_y, path_z = [], [], []
    z_hi, z_lo = int(max(z0, z1)), int(min(z0, z1))
    span = abs(z0 - z1) + 1e-6

    for z in range(z_hi, z_lo - 1, -1):
        t = abs(z - z0) / span
        gx = x0 + t * (x1 - x0)
        gy = y0 + t * (y1 - y0)

        sl = edt[:, :, z]
        r = search_radius
        xa, xb = max(0, int(gx - r)), min(sl.shape[0], int(gx + r + 1))
        ya, yb = max(0, int(gy - r)), min(sl.shape[1], int(gy + r + 1))
        win = sl[xa:xb, ya:yb]

        if np.max(win) > 0:
            mi = np.unravel_index(np.argmax(win), win.shape)
            path_x.append(xa + mi[0])
            path_y.append(ya + mi[1])
        else:
            path_x.append(gx)
            path_y.append(gy)
        path_z.append(z)

    return gaussian_filter1d(path_x, sigma=sigma), gaussian_filter1d(path_y, sigma=sigma), path_z


# ==========================================
# GPU: 3D Gaussian smoothing (separable)
# ==========================================
def gaussian_smooth_3d_gpu(vol, sigma):
    """Separable 3D Gaussian blur via three 1D convolutions on GPU."""
    ks = int(6 * sigma + 1) | 1  # kernel size, forced odd
    half = ks // 2
    t = torch.arange(ks, device=vol.device, dtype=torch.float32) - half
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = g / g.sum()

    v = vol.unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)

    # F.pad order for 5D: (Z_l, Z_r, Y_l, Y_r, X_l, X_r)
    v = F.conv3d(F.pad(v, (0, 0, 0, 0, half, half), mode='replicate'), g.view(1, 1, -1, 1, 1))
    v = F.conv3d(F.pad(v, (0, 0, half, half, 0, 0), mode='replicate'), g.view(1, 1, 1, -1, 1))
    v = F.conv3d(F.pad(v, (half, half, 0, 0, 0, 0), mode='replicate'), g.view(1, 1, 1, 1, -1))

    return v.squeeze(0).squeeze(0)


# ==========================================
# GPU: Build tube volumes
# ==========================================
def build_tube_gpu(shape, px, py, pz, diameter, thickness, shave_sigma):
    """Build solid + hollow tube entirely on GPU."""
    radius_sq = (diameter / 2.0) ** 2

    # XY coordinate grids (reused every slice)
    xg = torch.arange(shape[0], device=DEVICE, dtype=torch.float32).view(-1, 1)
    yg = torch.arange(shape[1], device=DEVICE, dtype=torch.float32).view(1, -1)

    # --- 1. Stamp circles along the path ---
    solid = torch.zeros(shape, device=DEVICE, dtype=torch.float32)
    for i, z in enumerate(pz):
        dist_sq = (xg - px[i]) ** 2 + (yg - py[i]) ** 2
        solid[:, :, z][dist_sq <= radius_sq] = 1.0

    # --- 2. Gaussian smooth → threshold (removes voxel staircase) ---
    solid = (gaussian_smooth_3d_gpu(solid, shave_sigma) >= 0.5).float()

    # --- 3. Hollow via batched 2D morphological erosion ---
    #     Replaces the per-slice EDT loop with a single conv2d over all Z.
    r = int(np.ceil(thickness))
    d = 2 * r + 1
    cy, cx = torch.meshgrid(
        torch.arange(d, device=DEVICE) - r,
        torch.arange(d, device=DEVICE) - r, indexing='ij')
    disk = (cx.float() ** 2 + cy.float() ** 2 <= thickness ** 2).float()
    disk_kernel = disk.unsqueeze(0).unsqueeze(0)   # (1, 1, d, d)
    disk_area = disk.sum()

    # Treat Z as batch dim → one conv2d call for ALL slices
    slices = solid.permute(2, 0, 1).unsqueeze(1)   # (Z, 1, X, Y)
    eroded = (F.conv2d(slices, disk_kernel, padding=r) >= disk_area).float()
    shell = ((slices > 0) & (eroded < 1)).float()   # thin wall only

    hollow = shell.squeeze(1).permute(1, 2, 0) * TUBE_INTENSITY  # (X, Y, Z)

    return solid.cpu().numpy().astype(np.uint8), hollow.cpu().numpy().astype(np.float32)


# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs("./output", exist_ok=True)

    # Load CT + segmentation
    ct_img = nib.load(INPUT_CT_PATH)
    ct_data = ct_img.get_fdata()
    mask = nib.load(generate_full_airway_mask(INPUT_CT_PATH, SEGS_DIR)).get_fdata().astype(np.uint8)

    # 1. Pick random endpoints
    print(f"Sampling endpoints (block={BLOCK_SIZE}, placement={TUBE_PLACEMENT})...")
    top, bot, placement = find_endpoints(mask, TUBE_HEIGHT_FRACTION, TUBE_PLACEMENT, BLOCK_SIZE)
    print(f"  Top:    {top}")
    print(f"  Bottom: {bot} ({placement})")

    # 2. Trace centerline (EDT on GPU via CuPy)
    print("Extracting safe centerline (GPU EDT)...")
    px, py, pz = extract_centerline(mask, top, bot, PATH_SEARCH_RADIUS, PATH_SMOOTHING_SIGMA)

    # 3. Build tube on GPU
    print(f"Building tube on {DEVICE}...")
    path_solid, tube_hollow = build_tube_gpu(
        ct_img.shape, px, py, pz, TUBE_DIAMETER, TUBE_THICKNESS, SHAVING_SIGMA)

    # 4. Sanity check: clip tube to segmentation
    tube_hollow[mask == 0] = 0

    # 5. Save NIfTI
    print("Saving NIfTI volumes...")
    aff, hdr = ct_img.affine, ct_img.header
    nib.save(nib.Nifti1Image(path_solid, aff, hdr), OUTPUT_PATH_NII)
    nib.save(nib.Nifti1Image(tube_hollow.astype(np.int16), aff, hdr), OUTPUT_TUBE_NII)

    # 6. Generate DRRs
    print("Generating DRR projections...")
    drr_save(path_solid.astype(np.float32), OUTPUT_PATH_DRR, multiplier=1000.0)
    drr_save(tube_hollow, OUTPUT_TUBE_DRR)

    combined = np.copy(ct_data)
    combined[tube_hollow > 0] = TUBE_INTENSITY
    drr_save(combined, OUTPUT_COMBINED_DRR)

    print("Done.")


if __name__ == "__main__":
    main()