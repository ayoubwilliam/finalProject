import numpy as np
import nibabel as nib
import torch
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, gaussian_filter

# Import our custom segmentation engine from seg.py
from seg import generate_full_airway_mask

# Import your DRR module and config
import config as cfg
import drr

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
INPUT_CT_PATH = "../ct/ct_file1.nii.gz"
SEGS_DIR = "./segs"

# Outputs (2x CT, 3x DRR)
OUTPUT_PATH_NII = "./output/curved_path_solid.nii.gz"
OUTPUT_TUBE_NII = "./output/curved_tube_hollow.nii.gz"

OUTPUT_PATH_DRR = "./output/drr_curved_path_solid.png"
OUTPUT_TUBE_DRR = "./output/drr_curved_tube_hollow.png"
OUTPUT_COMBINED_DRR = "./output/drr_ct_with_tube.png"

# ==========================================
# HYPERPARAMETERS
# ==========================================
# Tube Placement Options
TUBE_PLACEMENT = "RANDOM"  # Options: "MAIN", "LEFT", "RIGHT", "RANDOM"
TUBE_HEIGHT_FRACTION = 0.9  # Defines how far down the trachea the tube goes
ANCHOR_BLOCK_SIZE = 30  # Searches across this many Z-slices to find the absolute safest anchor

# Tube Geometry
TUBE_DIAMETER = 12.0  # Outer diameter of the tube in voxels
TUBE_THICKNESS = 3.0  # Wall thickness of the hollow tube in voxels
TUBE_INTENSITY = 1000.0  # HU value for the tube

# Physical Path Parameters
PATH_SEARCH_RADIUS = 15  # How far to look off the guide line for the safe center
PATH_SMOOTHING_SIGMA = 20.0  # Higher = stiffer/straighter tube path

# Shaving Parameters
SHAVING_SIGMA = 2.5  # Melts the sharp voxel edges. 1.0 - 3.5 is usually perfect.


# ==========================================
# 2. ENDPOINT LOGIC (MULTI-SLICE BLOCK)
# ==========================================
def get_safest_point_in_block(block_mask, z_start_idx):
    """
    Scans a 3D block of slices and finds the absolute widest,
    safest center point across all of them.
    """
    best_radius = -1
    best_point = None

    for z_local in range(block_mask.shape[2]):
        slice_mask = block_mask[:, :, z_local]
        if np.any(slice_mask):
            edt = distance_transform_edt(slice_mask)
            max_idx = np.unravel_index(np.argmax(edt), edt.shape)
            max_val = edt[max_idx]

            # If this slice has a wider safe zone, update our anchor
            if max_val > best_radius:
                best_radius = max_val
                best_point = (max_idx[0], max_idx[1], z_start_idx + z_local)

    if best_point is None:
        raise ValueError("The selected block is completely empty!")

    return best_point


def find_segmentation_endpoints(mask, height_fraction, placement, block_size):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("Mask is empty! Cannot calculate path length.")

    z_top_full = coords[:, 2].max()
    z_bottom_full = coords[:, 2].min()

    mask_length = z_top_full - z_bottom_full
    z_bottom_target = int(z_top_full - (mask_length * height_fraction))
    z_bottom_target = max(z_bottom_full, z_bottom_target)

    # 50% Probability coin flip for Random Branching
    if placement == "RANDOM":
        placement = np.random.choice(["LEFT", "RIGHT"])
        print(f"    -> Randomized branch selection: {placement}")

    # ---------------------------------------------------------
    # TOP POINT (Search across a 30-slice block)
    # ---------------------------------------------------------
    z_top_min = max(0, z_top_full - block_size)
    top_block = mask[:, :, z_top_min:z_top_full + 1]
    top_x, top_y, top_z = get_safest_point_in_block(top_block, z_top_min)

    # ---------------------------------------------------------
    # BOTTOM POINT (Search across a 30-slice block around target)
    # ---------------------------------------------------------
    half_block = block_size // 2
    z_bot_min = max(0, z_bottom_target - half_block)
    z_bot_max = min(mask.shape[2], z_bottom_target + half_block)

    bot_block = np.copy(mask[:, :, z_bot_min:z_bot_max])

    if placement in ["LEFT", "RIGHT"]:
        slice_coords = np.argwhere(bot_block > 0)
        if len(slice_coords) > 0:
            x_midpoint = np.mean(slice_coords[:, 0])

            # Zero out the unwanted half across the entire 3D block
            if placement == "LEFT":
                bot_block[int(x_midpoint):, :, :] = 0
            elif placement == "RIGHT":
                bot_block[:int(x_midpoint), :, :] = 0

        # Safety fallback
        if len(np.argwhere(bot_block > 0)) == 0:
            print(f"    -> Warning: Could not isolate {placement} branch. Falling back to main mask.")
            bot_block = mask[:, :, z_bot_min:z_bot_max]

    bot_x, bot_y, bot_z = get_safest_point_in_block(bot_block, z_bot_min)

    return (top_x, top_y, top_z), (bot_x, bot_y, bot_z), placement


# ==========================================
# 3. CONTINUOUS SAFE PATH EXTRACTION
# ==========================================
def extract_safe_centerline(mask, pt_top, pt_bot, search_radius, smoothing_sigma):
    edt = distance_transform_edt(mask)
    x_top, y_top, z_top = pt_top
    x_bot, y_bot, z_bot = pt_bot

    path_x, path_y, path_z = [], [], []

    z_start = int(max(z_top, z_bot))
    z_end = int(min(z_top, z_bot))

    for z in range(z_start, z_end - 1, -1):
        t = abs(z - z_top) / (abs(z_top - z_bot) + 1e-6)

        guide_x = x_top + t * (x_bot - x_top)
        guide_y = y_top + t * (y_bot - y_top)

        slice_edt = edt[:, :, z]

        x_min = max(0, int(guide_x - search_radius))
        x_max = min(slice_edt.shape[0], int(guide_x + search_radius + 1))
        y_min = max(0, int(guide_y - search_radius))
        y_max = min(slice_edt.shape[1], int(guide_y + search_radius + 1))

        local_window = slice_edt[x_min:x_max, y_min:y_max]

        if np.max(local_window) > 0:
            local_max_idx = np.unravel_index(np.argmax(local_window), local_window.shape)
            best_x = x_min + local_max_idx[0]
            best_y = y_min + local_max_idx[1]
        else:
            best_x, best_y = guide_x, guide_y

        path_x.append(best_x)
        path_y.append(best_y)
        path_z.append(z)

    smooth_x = gaussian_filter1d(path_x, sigma=smoothing_sigma)
    smooth_y = gaussian_filter1d(path_y, sigma=smoothing_sigma)

    return smooth_x, smooth_y, path_z


# ==========================================
# 4. 3D SHAPE GENERATION & LATE HOLLOWING
# ==========================================
def create_volumes_from_path(shape, path_x, path_y, path_z, tube_diameter, thickness, shaving_sigma):
    raw_tube_solid = np.zeros(shape, dtype=np.float32)
    tube_outer_radius = tube_diameter / 2.0

    x_grid = np.arange(shape[0])[:, None]
    y_grid = np.arange(shape[1])[None, :]

    print("    -> Drawing raw solid voxel boundaries...")
    for i, z in enumerate(path_z):
        cx = path_x[i]
        cy = path_y[i]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        raw_tube_solid[:, :, z][dist <= tube_outer_radius] = 1.0

    print(f"    -> Shaving jagged edges in 3D (Sigma: {shaving_sigma})...")
    smooth_tube_solid = (gaussian_filter(raw_tube_solid, sigma=shaving_sigma) >= 0.5).astype(np.uint8)

    print(f"    -> Carving out hollow core (Thickness: {thickness})...")
    volume_hollow = np.zeros(shape, dtype=np.float32)

    z_indices = np.where(np.any(smooth_tube_solid, axis=(0, 1)))[0]
    for z in z_indices:
        edt_slice = distance_transform_edt(smooth_tube_solid[:, :, z])
        hollow_mask = (smooth_tube_solid[:, :, z] > 0) & (edt_slice <= thickness)
        volume_hollow[:, :, z][hollow_mask] = TUBE_INTENSITY

    return smooth_tube_solid, volume_hollow


# ==========================================
# 5. DRR GENERATION
# ==========================================
def generate_drr_from_numpy(volume_np, output_filename, intensity_multiplier=1.0):
    print(f"  -> Projecting DRR to {output_filename}...")
    volume_scaled = volume_np * intensity_multiplier
    tensor = torch.from_numpy(volume_scaled).float().to(cfg.DEVICE)
    base_drr = drr.create_drr_from_ct(tensor)
    final_drr = drr.apply_drr_post_processing(base_drr)
    drr.save_drr(final_drr, output_filename)


# ==========================================
# 6. MAIN ORCHESTRATOR
# ==========================================
def main():
    os.makedirs("./output", exist_ok=True)

    print(f"Loading Original CT: {INPUT_CT_PATH}")
    ct_img = nib.load(INPUT_CT_PATH)
    ct_data = ct_img.get_fdata()
    ct_shape = ct_img.shape
    ct_affine = ct_img.affine

    # 1. Get Merged Segmentation
    mask_path = generate_full_airway_mask(INPUT_CT_PATH, SEGS_DIR)
    merged_mask = nib.load(mask_path).get_fdata()

    # 2. Find Robust Bounded Points safely inside the walls using block searching
    print(f"Finding safe anchoring points (Scanning {ANCHOR_BLOCK_SIZE} slices)...")
    pt_top, pt_bottom, actual_placement = find_segmentation_endpoints(
        merged_mask, TUBE_HEIGHT_FRACTION, TUBE_PLACEMENT, ANCHOR_BLOCK_SIZE
    )
    print(f"    -> Block-Safe Top: X={pt_top[0]:.1f}, Y={pt_top[1]:.1f}, Z={pt_top[2]}")
    print(
        f"    -> Block-Safe Bottom ({actual_placement}): X={pt_bottom[0]:.1f}, Y={pt_bottom[1]:.1f}, Z={pt_bottom[2]}")

    # 3. Extract the Continuous Safe Centerline
    print(f"Tracking anatomical center all the way down (Sigma: {PATH_SMOOTHING_SIGMA})...")
    path_x, path_y, path_z = extract_safe_centerline(
        mask=merged_mask,
        pt_top=pt_top,
        pt_bot=pt_bottom,
        search_radius=PATH_SEARCH_RADIUS,
        smoothing_sigma=PATH_SMOOTHING_SIGMA
    )

    # 4. Generate the 3D Volumes and Shave them smooth
    print(f"Extruding and shaving tube (Ø: {TUBE_DIAMETER})...")
    path_solid, tube_hollow = create_volumes_from_path(
        shape=ct_shape,
        path_x=path_x,
        path_y=path_y,
        path_z=path_z,
        tube_diameter=TUBE_DIAMETER,
        thickness=TUBE_THICKNESS,
        shaving_sigma=SHAVING_SIGMA
    )

    # 5. FINAL SAFETY CHECK: Strict mask against original anatomy
    print("Applying final anatomical bounds check (strict masking)...")
    tube_hollow[merged_mask == 0] = 0

    # 6. Save the 2x NIfTI CT volumes
    print("\n[Saving 3D CT Volumes]")
    print(f"Saving solid path NIfTI to: {OUTPUT_PATH_NII}")
    nib.save(nib.Nifti1Image(path_solid, ct_affine, ct_img.header), OUTPUT_PATH_NII)

    print(f"Saving hollow tube NIfTI to: {OUTPUT_TUBE_NII}")
    nib.save(nib.Nifti1Image(tube_hollow.astype(np.int16), ct_affine, ct_img.header), OUTPUT_TUBE_NII)

    # 7. Generate the 3x DRR Projections
    print("\n[Generating 2D DRR Projections]")
    generate_drr_from_numpy(path_solid, OUTPUT_PATH_DRR, intensity_multiplier=1000.0)
    generate_drr_from_numpy(tube_hollow, OUTPUT_TUBE_DRR, intensity_multiplier=1.0)

    print("  -> Merging hollow tube with original CT volume...")
    combined_ct = np.copy(ct_data)
    tube_voxels = tube_hollow > 0
    combined_ct[tube_voxels] = TUBE_INTENSITY
    generate_drr_from_numpy(combined_ct, OUTPUT_COMBINED_DRR, intensity_multiplier=1.0)

    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()