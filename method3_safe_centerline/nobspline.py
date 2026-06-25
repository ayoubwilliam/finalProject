import numpy as np
import nibabel as nib
import torch
import os

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
OUTPUT_COMBINED_DRR = "./output/drr_ct_with_tube.png"  # New output

# ==========================================
# HYPERPARAMETERS
# ==========================================
# Tube Geometry
PATH_DIAMETER = 10.0  # Outer diameter of the path/tube in voxels
TUBE_THICKNESS = 2.0  # Wall thickness of the hollow tube in voxels
TUBE_INTENSITY = 1000.0  # HU value for the tube
TUBE_HEIGHT_FRACTION = 0.9  # Defines how far down the trachea the path goes

# Parabolic Bend Parameters
MAX_BEND_COEFFICIENT = 2.0  # Max displacement in voxels at the apex of the curve
ASYMMETRY_COEFFICIENT = 2.0  # Shifts the apex slightly up or down the Z-axis


# ==========================================
# 2. ENDPOINT LOGIC
# ==========================================
def find_segmentation_endpoints(mask, height_fraction):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("Mask is empty! Cannot calculate path length.")

    z_top = coords[:, 2].max()
    z_bottom_full = coords[:, 2].min()

    mask_length = z_top - z_bottom_full
    z_bottom_target = int(z_top - (mask_length * height_fraction))
    z_bottom_target = max(z_bottom_full, z_bottom_target)

    # Top Centroid (Point A)
    top_coords = coords[coords[:, 2] == z_top]
    top_x = np.mean(top_coords[:, 0])
    top_y = np.mean(top_coords[:, 1])

    # Bottom Target Centroid (Point B)
    bottom_coords = coords[coords[:, 2] == z_bottom_target]
    if len(bottom_coords) == 0:
        closest_z = coords[np.argmin(np.abs(coords[:, 2] - z_bottom_target)), 2]
        bottom_coords = coords[coords[:, 2] == closest_z]
        z_bottom_target = closest_z

    bottom_x = np.mean(bottom_coords[:, 0])
    bottom_y = np.mean(bottom_coords[:, 1])

    return (top_x, top_y, z_top), (bottom_x, bottom_y, z_bottom_target)


# ==========================================
# 3. 3D SHAPE GENERATION
# ==========================================
def create_curved_volumes(shape, pt_start, pt_end, diameter, thickness, max_bend_coeff, asymmetry_coeff):
    """
    Simultaneously draws the solid path mask and the hollow tube mask to save compute time.
    """
    volume_solid = np.zeros(shape, dtype=np.uint8)
    volume_hollow = np.zeros(shape, dtype=np.float32)

    radius = diameter / 2.0

    x_top, y_top, z_top = pt_start
    x_bot, y_bot, z_bot = pt_end

    x_grid = np.arange(shape[0])[:, None]
    y_grid = np.arange(shape[1])[None, :]

    z_min = min(int(z_bot), int(z_top))
    z_max = max(int(z_bot), int(z_top))

    for z in range(z_min, z_max + 1):
        # Normalized progression from Top (0.0) to Bottom (1.0)
        t = abs(z - z_top) / (abs(z_top - z_bot) + 1e-6)

        # 1. Calculate straight line + parabolic bend offset
        line_x = x_top + t * (x_bot - x_top)
        line_y = y_top + t * (y_bot - y_top)

        base_parabola = 4.0 * max_bend_coeff * t * (1.0 - t)
        asymmetry_factor = 1.0 + asymmetry_coeff * (t - 0.5)
        bend_offset = base_parabola * asymmetry_factor

        cx = line_x + bend_offset
        cy = line_y

        # 2. Distance from the curved center
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        # 3. Draw Solid Path
        volume_solid[:, :, z][dist <= radius] = 1

        # 4. Draw Hollow Tube
        ring_mask = (dist <= radius) & (dist >= (radius - thickness))
        volume_hollow[:, :, z][ring_mask] = TUBE_INTENSITY

    return volume_solid, volume_hollow


# ==========================================
# 4. DRR GENERATION
# ==========================================
def generate_drr_from_numpy(volume_np, output_filename, intensity_multiplier=1.0):
    """
    Takes a generated numpy volume, scales it, passes it through the PyTorch DRR pipeline, and saves.
    """
    print(f"  -> Projecting DRR to {output_filename}...")

    # Scale and convert to tensor
    volume_scaled = volume_np * intensity_multiplier
    tensor = torch.from_numpy(volume_scaled).float().to(cfg.DEVICE)

    # DRR Pipeline
    base_drr = drr.create_drr_from_ct(tensor)
    final_drr = drr.apply_drr_post_processing(base_drr)

    # Save Output
    drr.save_drr(final_drr, output_filename)


# ==========================================
# 5. MAIN ORCHESTRATOR
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

    # 2. Find Point A (Top) and Point B (Bottom)
    pt_top, pt_bottom = find_segmentation_endpoints(merged_mask, TUBE_HEIGHT_FRACTION)
    print(f"Path Start (Top): X={pt_top[0]:.1f}, Y={pt_top[1]:.1f}, Z={pt_top[2]}")
    print(f"Path End (Bottom): X={pt_bottom[0]:.1f}, Y={pt_bottom[1]:.1f}, Z={pt_bottom[2]}")

    # 3. Generate the 3D Math Curves (Solid and Hollow)
    print(f"Drawing curved shapes (Diameter: {PATH_DIAMETER}, Max Bend: {MAX_BEND_COEFFICIENT} voxels)...")
    path_solid, tube_hollow = create_curved_volumes(
        shape=ct_shape,
        pt_start=pt_top,
        pt_end=pt_bottom,
        diameter=PATH_DIAMETER,
        thickness=TUBE_THICKNESS,
        max_bend_coeff=MAX_BEND_COEFFICIENT,
        asymmetry_coeff=ASYMMETRY_COEFFICIENT
    )

    # 4. Save the 2x NIfTI CT volumes
    print("\n[Saving 3D CT Volumes]")
    print(f"Saving solid path NIfTI to: {OUTPUT_PATH_NII}")
    nib.save(nib.Nifti1Image(path_solid, ct_affine, ct_img.header), OUTPUT_PATH_NII)

    print(f"Saving hollow tube NIfTI to: {OUTPUT_TUBE_NII}")
    nib.save(nib.Nifti1Image(tube_hollow.astype(np.int16), ct_affine, ct_img.header), OUTPUT_TUBE_NII)

    # 5. Generate the 3x DRR Projections
    print("\n[Generating 2D DRR Projections]")

    # DRR 1: The solid path (multiply by 1000.0 to simulate density)
    generate_drr_from_numpy(path_solid, OUTPUT_PATH_DRR, intensity_multiplier=1000.0)

    # DRR 2: The hollow tube (already 1000 HU)
    generate_drr_from_numpy(tube_hollow, OUTPUT_TUBE_DRR, intensity_multiplier=1.0)

    # DRR 3: The Combined CT + Tube
    print("  -> Merging hollow tube with original CT volume...")
    combined_ct = np.copy(ct_data)
    tube_voxels = tube_hollow > 0

    # Overwrite the original CT voxels where the tube exists
    combined_ct[tube_voxels] = TUBE_INTENSITY
    generate_drr_from_numpy(combined_ct, OUTPUT_COMBINED_DRR, intensity_multiplier=1.0)

    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()