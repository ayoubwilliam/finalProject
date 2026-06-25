import numpy as np
import nibabel as nib
import os
import sys

# Import our custom segmentation engine from seg.py
from seg import generate_full_airway_mask

# Move the heavy imports inside the main guard or functions where possible
# to keep Windows child processes lightweight
import gryds

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
INPUT_CT_PATH = "../ct/ct_file1.nii.gz"
OUTPUT_TUBE_PATH = "./output/generated_banana_tube_cuda.nii.gz"
SEGS_DIR = "./segs"

# Tube parameters
TUBE_RADIUS = 4.0  # Outer radius in voxels
TUBE_THICKNESS = 2.0  # Wall thickness in voxels
TUBE_INTENSITY = 1000.0  # HU value for the tube

# Defines how far down the trachea the tube goes (1.0 = full length, 0.7 = 70% down)
TUBE_HEIGHT_FRACTION = 0.7

# Deformation parameters
GRID_RESOLUTION = (30, 30, 70)
MAX_BEND_COEFFICIENT = -0.1
ASYMMETRY_COEFFICIENT = 1.0


# ==========================================
# 2. BOUNDING LOGIC
# ==========================================
def analyze_bounds(mask, height_fraction):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("Mask is empty! Cannot place tube.")

    z_start_full = coords[:, 2].min()
    z_end = coords[:, 2].max()

    # Calculate tube length based on the fraction and anchor it to the top (z_end)
    mask_length = z_end - z_start_full
    z_start = int(z_end - (mask_length * height_fraction))

    # Safety clamp to ensure we don't go out of bounds
    z_start = max(z_start_full, z_start)

    # Find all points at the "top" slice of the mask (highest Z value)
    top_slice_coords = coords[coords[:, 2] == z_end]

    # Pick a random pixel from the top slice to be the center
    random_index = np.random.randint(0, len(top_slice_coords))
    cx = int(top_slice_coords[random_index, 0])
    cy = int(top_slice_coords[random_index, 1])

    return (cx, cy), int(z_start), int(z_end)


# ==========================================
# 3. SHAPE GENERATION
# ==========================================
def create_hollow_tube(shape, center_xy, radius, thickness, z_start, z_end):
    volume = np.zeros(shape, dtype=np.float32)
    cx, cy = center_xy

    x = np.arange(shape[0])[:, None]
    y = np.arange(shape[1])[None, :]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    ring_mask = (dist <= radius) & (dist >= (radius - thickness))
    volume[ring_mask, :] = TUBE_INTENSITY

    volume[:, :, :z_start] = 0
    volume[:, :, z_end:] = 0

    return volume


# ==========================================
# 4. STRUCTURED DEFORMATION (CUDA)
# ==========================================
def apply_banana_bend_cuda(volume_data, grid_res, max_bend_coeff, asymmetry_coeff):
    nx, ny, nz = grid_res

    gridx = np.zeros((nx, ny, nz), dtype=np.float32)
    gridy = np.zeros((nx, ny, nz), dtype=np.float32)
    gridz = np.zeros((nx, ny, nz), dtype=np.float32)

    for k in range(nz):
        z_norm = k / (nz - 1)
        base_parabola = 4.0 * max_bend_coeff * z_norm * (1.0 - z_norm)
        asymmetry_factor = 1.0 + asymmetry_coeff * (z_norm - 0.5)

        displacement = base_parabola * asymmetry_factor
        gridx[:, :, k] = displacement

    transform = gryds.BSplineTransformationCuda([gridx, gridy, gridz])
    interpolator = gryds.BSplineInterpolatorCuda(volume_data, order=1, mode="mirror")
    deformed_tube = interpolator.transform(transform)

    return deformed_tube


# ==========================================
# 5. MAIN ORCHESTRATOR
# ==========================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_TUBE_PATH), exist_ok=True)

    print(f"Loading Original CT: {INPUT_CT_PATH}")
    ct_img = nib.load(INPUT_CT_PATH)
    ct_shape = ct_img.shape
    ct_affine = ct_img.affine

    # 1. Get Merged Segmentation from seg.py
    mask_path = generate_full_airway_mask(INPUT_CT_PATH, SEGS_DIR)
    merged_mask = nib.load(mask_path).get_fdata()

    # 2. Calculate tube placement based on Merged Mask and height fraction
    center_xy, z_start, z_end = analyze_bounds(merged_mask, TUBE_HEIGHT_FRACTION)
    print(f"Bounds Found -> Random Top Center: {center_xy}, Z_Start: {z_start}, Z_End: {z_end}")

    # 3. Create straight tube
    print("Generating straight hollow tube starting from random top...")
    tube_volume = create_hollow_tube(ct_shape, center_xy, TUBE_RADIUS, TUBE_THICKNESS, z_start, z_end)

    # 4. Apply deformation
    print("Applying asymmetric structural banana curve via CUDA...")
    deformed_tube = apply_banana_bend_cuda(tube_volume, GRID_RESOLUTION, MAX_BEND_COEFFICIENT, ASYMMETRY_COEFFICIENT)

    # 5. Save the generated tube
    print(f"Saving generated tube to: {OUTPUT_TUBE_PATH}")
    out_banana = nib.Nifti1Image(deformed_tube.astype(np.int16), ct_affine, ct_img.header)
    nib.save(out_banana, OUTPUT_TUBE_PATH)

    print("Done!")


if __name__ == "__main__":
    # Windows-specific memory optimization for PyTorch allocation
    os.environ["PyTorch_GEOM_ALLOCATOR"] = "CUDA_MALLOC"
    main()