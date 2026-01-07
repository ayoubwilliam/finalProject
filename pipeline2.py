import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from file_handler import load_nifti, save_nifti
from deformed_mass_generator import get_deformed_sphere_fast
from pooling import apply_pooling
from rotation import rotate_ct_scan
from drr_with_post_processing import create_drr_from_ct, save_drr, apply_drr_post_processing
from constants import DEVICE

# numeric constants
POOLING_KERNEL_SIZE = 8
INTENSITY = 25
GRID_DENSITY_FACTOR = 16
DEFORMATION_FACTOR = 0.2

# images filenames
OUTPUT_DIR = "pipeline_output/"
CURRENT_FILENAME = "current.png"
PRIOR_BY_PRIOR_FILENAME = "prior_rotated_to_prior.png"
PRIOR_BY_CURRENT_FILENAME = "prior_rotated_to_current.png"
HEATMAP_FILENAME = "heatmap.png"

# ct filename
PRIOR_DEFORMED_MASK = "prior_bspline_fast_mask.nii.gz"
PRIOR_DEFORMED_MASS = "prior_bspline_fast.nii.gz"
PRIOR_POOLED_MASK = "prior_pooling.nii.gz"
CURRENT_DEFORMED_MASK = "current_bspline_fast_mask.nii.gz"
CURRENT_DEFORMED_MASS = "current_bspline_fast.nii.gz"
CURRENT_POOLED_MASK = "current_pooling.nii.gz"

colors = [
    (0, 1, 0, 1),  # Green (neg values)
    (0, 1, 0, 0),  # Transparent (Approaching 0 from positive)
    (1, 0, 0, 0),  # Transparent (Approaching 0 from negative)
    (1, 0, 0, 1)  # Red (pos values)
]
custom_cmap = LinearSegmentedColormap.from_list("RedClearGreen", colors, N=256)


def correct_mask_by_seg(mask, seg) -> np.ndarray:
    return mask.bool() & seg.bool()


def apply_mask(destination_data, source_data, mask) -> None:
    positive_mask = (destination_data < source_data) & mask
    destination_data[positive_mask] = source_data[positive_mask]


import torch


def add_mass(data, seg, pos, radius, margin, pair_dir, affine, header):
    # 1. Ensure inputs are on GPU for the fast generation step
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float().to(DEVICE)
    if isinstance(seg, np.ndarray):
        seg = torch.from_numpy(seg).bool().to(DEVICE)

    working_data = data.clone()
    print("Running add_deformed_sphere_fast...")

    # Pass the actual 'working_data' tensor, not just the shape
    deformed_sphere, mask = get_deformed_sphere_fast(working_data, INTENSITY, pos, radius, margin,
                                                     GRID_DENSITY_FACTOR, DEFORMATION_FACTOR)

    # Resume original logic (Numpy/CPU)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(working_data, deformed_sphere, mask)

    # save_nifti(pair_dir + PRIOR_DEFORMED_MASK, deformed_sphere, affine, header)
    # save_nifti(pair_dir + PRIOR_DEFORMED_MASS, working_data, affine, header)
    print("finished deformed_sphere_fast...")

    # apply pooling
    print("start pooling...")

    pooled_data, mask = apply_pooling(working_data, mask, POOLING_KERNEL_SIZE)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(working_data, pooled_data, mask)

    # save_nifti(pair_dir + PRIOR_POOLED_MASK, working_data, affine, header)

    print("finished pooling.")
    return working_data, mask


def create_prior_ct(prior: np.ndarray, seg: np.ndarray,
                    prior_pos: tuple[int, int, int], radius: int, margin: int,
                    affine: np.ndarray, header: nib.Nifti1Header, pair_dir: str):
    # create deformed mass
    working_data, mask = add_mass(prior, seg, prior_pos, radius, margin, pair_dir, affine, header)
    apply_mask(prior, working_data, mask)
    return prior


def create_current_ct(current: np.ndarray, seg: np.ndarray,
                      current_pos: tuple[int, int, int], radius: int, margin: int,
                      affine: np.ndarray, header: nib.Nifti1Header, pair_dir: str):
    working_data, mask = add_mass(current, seg, current_pos, radius, margin, pair_dir, affine, header)
    apply_mask(current, working_data, mask)
    return current


def rotate_and_drr(data: np.ndarray, angles: tuple[float, float, float]) -> np.ndarray:
    rotated_ct = rotate_ct_scan(data, angles[0], angles[1], angles[2])
    # rotated_ct = rotated_ct.detach().cpu().numpy()
    drr = create_drr_from_ct(rotated_ct)
    # save_drr(drr, output_filename)
    return drr


def create_heatmap(current_drr, current_pp, prior_rotated_to_current_drr,
                   heatmap_path: str) -> None:
    # Ensure data is on CPU and Numpy ---
    def ensure_numpy(data):
        if isinstance(data, torch.Tensor):
            # Detach from graph, move to CPU, convert to numpy
            return data.detach().cpu().numpy()
        # If it's already numpy (or list), just ensure it's an array
        return np.asarray(data)

    # 1. Convert ALL inputs to Numpy
    current_drr = ensure_numpy(current_drr)
    prior_rotated_to_current_drr = ensure_numpy(prior_rotated_to_current_drr)
    current_pp = ensure_numpy(current_pp)  # <--- This was the one causing your specific error

    # 2. Calculate the difference
    heatmap = current_drr - prior_rotated_to_current_drr

    # 3. Create a figure with a single axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Background (Current Image)
    ax.imshow(current_pp, cmap='gray')

    # Plot Heatmap Overlay
    max_error = np.max(np.abs(heatmap))
    im = ax.imshow(heatmap, cmap=custom_cmap, alpha=1, vmin=-max_error, vmax=max_error)

    # Styling
    ax.set_title("Difference Heatmap")
    ax.axis('off')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    # Save the figure
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)

    # Close the plot to free up memory
    plt.close(fig)


def get_filename_from_path(path: str) -> str:
    return path.split('/')[-1].split('.')[0]


def get_pair_dir(pair_index: int, input_path: str) -> str:
    input_filename = get_filename_from_path(input_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = OUTPUT_DIR + input_filename + "/Pair" + str(pair_index) + "/"
    os.makedirs(path, exist_ok=True)  # Creates the folder if it doesn't exist
    return path


def pipeline(pair_index: int, input_path: str, seg_path: str, radius: int,
             prior_pos: tuple[int, int, int], current_pos: tuple[int, int, int],
             prior_angles: tuple[float, float, float], current_angles: tuple[float, float, float]) -> None:
    print("loading data")
    data_np, affine, header = load_nifti(input_path)
    seg_np, _, _ = load_nifti(seg_path)

    # --- MOVE TO GPU ---
    # 'data' acts as our read-only template. We keep it until the very end.
    data = torch.from_numpy(data_np).float().to(DEVICE)
    seg = torch.from_numpy(seg_np).bool().to(DEVICE)

    print("finished loading data")

    margin = radius
    pair_dir = get_pair_dir(pair_index, input_path)

    # ==========================================
    # PHASE 1: Process Current CT
    # ==========================================
    # 1. Create a copy for 'current'
    print("current: ")
    current_data = data.clone()

    # 2. Add mass (GPU)
    current_data = create_current_ct(current_data, seg,
                                     current_pos, radius, margin,
                                     affine, header, pair_dir)

    # 3. Generate DRR (Heavy GPU Operation)
    # The result 'current_drr' should be a small 2D image (CPU/Numpy)
    current_drr = rotate_and_drr(current_data, current_angles)

    # 4. CRITICAL: Delete 'current_data' immediately to free VRAM
    del current_data
    torch.cuda.empty_cache()  # Force PyTorch to release the memory for the next step

    # ==========================================
    # PHASE 2: Process Prior CT
    # ==========================================
    # 1. Create a copy for 'prior' (Now we have space again!)
    print("prior: ")
    prior_data = data.clone()

    # 2. Add mass (GPU)
    prior_data = create_prior_ct(prior_data, seg,
                                 prior_pos, radius, margin,
                                 affine, header, pair_dir)

    # 3. Generate DRRs
    # We use the same 'prior_data' 3D volume for both rotations.
    # We do NOT need to clone() it again here, saving another chunk of VRAM.
    prior_rotated_to_current_drr = rotate_and_drr(prior_data, current_angles)
    prior_rotated_to_prior_drr = rotate_and_drr(prior_data, prior_angles)

    # 4. CRITICAL: Delete 'prior_data' and original 'data'
    del prior_data
    del data
    del seg
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE 3: Post-Processing (CPU / Lightweight)
    # ==========================================
    # At this point, VRAM is empty. We only have the small 2D DRR images on CPU.
    print("post processing and drr...")

    # pp and save drr
    current_pp = apply_drr_post_processing(current_drr)
    save_drr(current_pp, pair_dir + CURRENT_FILENAME)

    prior_by_prior_pp = apply_drr_post_processing(prior_rotated_to_prior_drr)
    save_drr(prior_by_prior_pp, pair_dir + PRIOR_BY_PRIOR_FILENAME)

    prior_by_current_pp = apply_drr_post_processing(prior_rotated_to_current_drr)
    save_drr(prior_by_current_pp, pair_dir + PRIOR_BY_CURRENT_FILENAME)

    # create heatmap
    print("heatmap...")
    create_heatmap(current_drr, current_pp, prior_rotated_to_current_drr, pair_dir + HEATMAP_FILENAME)

    print("Done!")
