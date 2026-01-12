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
POOLING_KERNEL_SIZE = 13
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


from matplotlib.colors import LinearSegmentedColormap

colors = [
    # --- Negative Side (Green) ---
    (0.0,  (0, 1, 0, 1.0)),  # -1.0 : Green, Opaque
    (0.5,  (0, 1, 0, 0.0)),  #  0.0 : Green, Transparent

    # --- Positive Side (Red) ---
    (0.5,  (1, 0, 0, 0.0)),  #  0.0 : Red,   Transparent (Instant Color Switch)
    (1.0,  (1, 0, 0, 1.0))   # +1.0 : Red,   Opaque
]

custom_cmap = LinearSegmentedColormap.from_list("HardSwitch_SmoothAlpha", colors)
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

    # FIX 1: Pass the actual 'working_data' tensor, not just the shape
    deformed_sphere, mask = get_deformed_sphere_fast(working_data, INTENSITY, pos, radius, margin,
                                                     GRID_DENSITY_FACTOR, DEFORMATION_FACTOR)

    # --- FIX 2: Transfer to CPU/Numpy for the rest of the function ---
    # Since the subsequent lines use 'save_nifti' and likely Numpy-based helper functions,
    # we convert everything back to Numpy here as requested.


    # --- Resume original logic (Numpy/CPU) ---
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
    working_data, mask = add_mass(prior, seg, prior_pos, radius, margin,pair_dir,affine,header)
    apply_mask(prior, working_data, mask)
    return prior


def create_current_ct(current: np.ndarray, seg: np.ndarray,
                      current_pos: tuple[int, int, int], radius: int, margin: int,
                      affine: np.ndarray, header: nib.Nifti1Header, pair_dir: str):
    working_data, mask = add_mass(current, seg, current_pos, radius, margin,pair_dir,affine,header)
    apply_mask(current, working_data, mask)
    return current




from crop import get_segmentation_bounds
def rotate_and_drr(data: np.ndarray, angles: tuple[float, float, float], seg, crop_margin=10) -> np.ndarray:

    rotated_ct_gpu = rotate_ct_scan(data, angles[0], angles[1], angles[2])

    #clear gpu memory
    rotated_ct_cpu = rotated_ct_gpu.detach().cpu()
    del rotated_ct_gpu
    torch.cuda.empty_cache()


    bounds = get_segmentation_bounds(seg, angles, crop_margin)


    # Clear cache again just to be safe after segmentation work
    torch.cuda.empty_cache()

    # --- STEP 3: Apply Crop and Reload to GPU ---
    if bounds is None:
        print("Warning: Empty segmentation. Using full rotated volume.")
        # If seg is empty, we must send the whole thing back (RISK of OOM, but unavoidable)
        cropped_ct_gpu = rotated_ct_cpu.to(DEVICE)
    else:
        z1, z2, y1, y2, x1, x2 = bounds

        # Apply crop on the CPU tensor first (Fast and Memory Safe)
        cropped_ct_cpu = rotated_ct_cpu[z1:z2, y1:y2, x1:x2]

        # Reload ONLY the cropped portion into GPU memory
        cropped_ct_gpu = cropped_ct_cpu.to(DEVICE)

    # Clean up the large CPU tensor
    del rotated_ct_cpu

    # --- STEP 4: Create DRR ---
    drr = create_drr_from_ct(cropped_ct_gpu)

    return drr


import os
from file_handler import save_image_as_nifti, load_image_from_nifti
def create_heatmap(current_drr, current_pp, prior_rotated_to_current_drr,
                   heatmap_path: str) -> None:
    def ensure_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    current_drr = ensure_numpy(current_drr)
    prior_rotated_to_current_drr = ensure_numpy(prior_rotated_to_current_drr)
    current_pp = ensure_numpy(current_pp)

    heatmap = current_drr - prior_rotated_to_current_drr

    max_error = np.max(np.abs(heatmap))

    base, ext = os.path.splitext(heatmap_path)
    overlay_path = f"{base}_overlay{ext}"

    # --- 1. Save WITH Background (Overlay) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(current_pp, cmap='gray')
    im = ax.imshow(heatmap, cmap=custom_cmap, alpha=1, vmin=-max_error, vmax=max_error)
    ax.set_title("Difference Heatmap (Overlay)")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    print(f"Saving overlay heatmap to {overlay_path}")
    plt.savefig(overlay_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # --- 2. Save WITHOUT Background (Heatmap Only) ---
    save_image_as_nifti(heatmap, base + ".nii.gz")


def get_filename_from_path(path: str) -> str:
    return path.split('/')[-1].split('.')[0]


def get_pair_dir(pair_index: int, input_path: str) -> str:
    input_filename = get_filename_from_path(input_path)
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
    current_data = data.clone()

    # 2. Add mass (GPU)
    current_data = create_current_ct(current_data, seg,
                                     current_pos, radius, margin,
                                     affine, header, pair_dir)

    # 3. Generate DRR (Heavy GPU Operation)
    # The result 'current_drr' should be a small 2D image (CPU/Numpy)
    current_drr = rotate_and_drr(current_data, current_angles,seg)

    # 4. CRITICAL: Delete 'current_data' immediately to free VRAM
    del current_data
    torch.cuda.empty_cache()  # Force PyTorch to release the memory for the next step

    # ==========================================
    # PHASE 2: Process Prior CT
    # ==========================================
    # 1. Create a copy for 'prior' (Now we have space again!)
    prior_data = data.clone()

    # 2. Add mass (GPU)
    prior_data = create_prior_ct(prior_data, seg,
                                 prior_pos, radius, margin,
                                 affine, header, pair_dir)

    # 3. Generate DRRs
    # We use the same 'prior_data' 3D volume for both rotations.
    # We do NOT need to clone() it again here, saving another chunk of VRAM.
    prior_rotated_to_current_drr = rotate_and_drr(prior_data, current_angles,seg)
    prior_rotated_to_prior_drr = rotate_and_drr(prior_data, prior_angles,seg)

    # 4. CRITICAL: Delete 'prior_data' and original 'data'
    del prior_data
    del data
    del seg
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE 3: Post-Processing (CPU / Lightweight)
    # ==========================================
    # At this point, VRAM is empty. We only have the small 2D DRR images on CPU.

    # pp and save drr
    current_pp = apply_drr_post_processing(current_drr)
    save_image_as_nifti(current_pp.cpu().numpy(), pair_dir + "current.nii.gz")
    save_drr(current_pp, pair_dir + CURRENT_FILENAME)

    prior_by_prior_pp = apply_drr_post_processing(prior_rotated_to_prior_drr)
    save_image_as_nifti(prior_by_prior_pp.cpu().numpy(), pair_dir + "prior.nii.gz")
    save_drr(prior_by_prior_pp, pair_dir + PRIOR_BY_PRIOR_FILENAME)

    prior_by_current_pp = apply_drr_post_processing(prior_rotated_to_current_drr)
    save_drr(prior_by_current_pp, pair_dir + PRIOR_BY_CURRENT_FILENAME)

    # create heatmap
    create_heatmap(current_drr, current_pp, prior_rotated_to_current_drr, pair_dir + HEATMAP_FILENAME)

    print("Done!")

