import numpy as np
import nibabel as nib
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, gaussian_filter

# Import our custom segmentation engine from seg.py
from seg import generate_full_airway_mask

# Import your DRR module and config
import config as cfg
import drr
import torch

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
INPUT_CT_PATH = "../ct/ct_file1.nii.gz"
SEGS_DIR = "./segs"

# Standardized Output Names
OUTPUT_PATH_NII = "./output/curved_path_solid.nii.gz"
OUTPUT_TUBE_NII = "./output/curved_tube_hollow.nii.gz"

OUTPUT_PATH_DRR = "./output/drr_curved_path_solid.png"
OUTPUT_TUBE_DRR = "./output/drr_curved_tube_hollow.png"
OUTPUT_COMBINED_DRR = "./output/drr_ct_with_tube.png"

# ==========================================
# HYPERPARAMETERS
# ==========================================
TUBE_HEIGHT_FRACTION = 0.9
TUBE_DIAMETER = 8.0
TUBE_THICKNESS = 3.0
TUBE_INTENSITY = 1000.0

PATH_SEARCH_RADIUS = 15
PATH_SMOOTHING_SIGMA = 20.0
SHAVING_SIGMA = 2.5
ANCHOR_BLOCK_SIZE = 30


# ==========================================
# CORE LOGIC
# ==========================================
def get_safest_point_in_block(block_mask, z_start_idx):
    best_radius, best_point = -1, None
    for z_local in range(block_mask.shape[2]):
        slice_mask = block_mask[:, :, z_local]
        if np.any(slice_mask):
            edt = distance_transform_edt(slice_mask)
            max_idx = np.unravel_index(np.argmax(edt), edt.shape)
            if edt[max_idx] > best_radius:
                best_radius = edt[max_idx]
                best_point = (max_idx[0], max_idx[1], z_start_idx + z_local)
    return best_point


def find_segmentation_endpoints(mask, height_fraction, block_size):
    coords = np.argwhere(mask > 0)
    z_top, z_bot = coords[:, 2].max(), coords[:, 2].min()
    z_target = max(z_bot, int(z_top - (z_top - z_bot) * height_fraction))

    top_pt = get_safest_point_in_block(mask[:, :, max(0, z_top - block_size):z_top + 1], max(0, z_top - block_size))
    bot_pt = get_safest_point_in_block(
        mask[:, :, max(0, z_target - block_size // 2):min(mask.shape[2], z_target + block_size // 2)],
        max(0, z_target - block_size // 2))
    return top_pt, bot_pt


def extract_safe_centerline(mask, pt_top, pt_bot, search_radius, smoothing_sigma):
    edt = distance_transform_edt(mask)
    x_top, y_top, z_top = pt_top
    x_bot, y_bot, z_bot = pt_bot
    path_x, path_y, path_z = [], [], []

    for z in range(int(max(z_top, z_bot)), int(min(z_top, z_bot)) - 1, -1):
        t = abs(z - z_top) / (abs(z_top - z_bot) + 1e-6)
        guide_x, guide_y = x_top + t * (x_bot - x_top), y_top + t * (y_bot - y_top)

        slice_edt = edt[:, :, z]
        x_min, x_max = max(0, int(guide_x - search_radius)), min(slice_edt.shape[0], int(guide_x + search_radius + 1))
        y_min, y_max = max(0, int(guide_y - search_radius)), min(slice_edt.shape[1], int(guide_y + search_radius + 1))

        local_window = slice_edt[x_min:x_max, y_min:y_max]
        if np.max(local_window) > 0:
            local_max = np.unravel_index(np.argmax(local_window), local_window.shape)
            path_x.append(x_min + local_max[0]);
            path_y.append(y_min + local_max[1])
        else:
            path_x.append(guide_x);
            path_y.append(guide_y)
        path_z.append(z)

    return gaussian_filter1d(path_x, sigma=smoothing_sigma), gaussian_filter1d(path_y, sigma=smoothing_sigma), path_z


def create_hollow_tube(shape, path_x, path_y, path_z, tube_diameter, thickness, shaving_sigma):
    raw_solid = np.zeros(shape, dtype=np.float32)
    radius = tube_diameter / 2.0
    x_grid, y_grid = np.arange(shape[0])[:, None], np.arange(shape[1])[None, :]

    for i, z in enumerate(path_z):
        dist = np.sqrt((x_grid - path_x[i]) ** 2 + (y_grid - path_y[i]) ** 2)
        raw_solid[:, :, z][dist <= radius] = 1.0

    smooth_solid = (gaussian_filter(raw_solid, sigma=shaving_sigma) >= 0.5).astype(np.uint8)

    volume_hollow = np.zeros(shape, dtype=np.float32)
    for z in np.where(np.any(smooth_solid, axis=(0, 1)))[0]:
        edt_slice = distance_transform_edt(smooth_solid[:, :, z])
        volume_hollow[:, :, z][(smooth_solid[:, :, z] > 0) & (edt_slice <= thickness)] = TUBE_INTENSITY
    return smooth_solid, volume_hollow


# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs("./output", exist_ok=True)
    ct_img = nib.load(INPUT_CT_PATH)
    merged_mask = nib.load(generate_full_airway_mask(INPUT_CT_PATH, SEGS_DIR)).get_fdata()

    pt_top, pt_bot = find_segmentation_endpoints(merged_mask, TUBE_HEIGHT_FRACTION, ANCHOR_BLOCK_SIZE)
    px, py, pz = extract_safe_centerline(merged_mask, pt_top, pt_bot, PATH_SEARCH_RADIUS, PATH_SMOOTHING_SIGMA)

    path_solid, tube_hollow = create_hollow_tube(ct_img.shape, px, py, pz, TUBE_DIAMETER, TUBE_THICKNESS, SHAVING_SIGMA)
    tube_hollow[merged_mask == 0] = 0

    # Save and DRR
    for vol, name in [(path_solid, OUTPUT_PATH_NII), (tube_hollow.astype(np.int16), OUTPUT_TUBE_NII)]:
        nib.save(nib.Nifti1Image(vol, ct_img.affine, ct_img.header), name)

    # Simplified DRR call
    for vol, name, mult in [(path_solid, OUTPUT_PATH_DRR, 1000.0), (tube_hollow, OUTPUT_TUBE_DRR, 1.0)]:
        drr.save_drr(
            drr.apply_drr_post_processing(drr.create_drr_from_ct(torch.from_numpy(vol * mult).float().to(cfg.DEVICE))),
            name)

    combined = np.copy(ct_img.get_fdata())
    combined[tube_hollow > 0] = TUBE_INTENSITY
    drr.save_drr(
        drr.apply_drr_post_processing(drr.create_drr_from_ct(torch.from_numpy(combined).float().to(cfg.DEVICE))),
        OUTPUT_COMBINED_DRR)


if __name__ == "__main__":
    main()