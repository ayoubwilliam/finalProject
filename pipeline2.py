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


def correct_mask_by_seg(mask: np.ndarray, seg: np.ndarray) -> np.ndarray:
    return mask.astype(bool) & seg.astype(bool)


def apply_mask(destination_data: np.ndarray, source_data: np.ndarray, mask: np.ndarray) -> None:
    positive_mask = (destination_data < source_data) & mask
    destination_data[positive_mask] = source_data[positive_mask]


def add_mass(data, seg, pos, radius, margin):
    working_data = data.copy()
    print("Running add_deformed_sphere_fast...")
    deformed_sphere, mask = get_deformed_sphere_fast(working_data.shape, INTENSITY, pos, radius, margin,
                                                     GRID_DENSITY_FACTOR, DEFORMATION_FACTOR)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(working_data, deformed_sphere, mask)
    # save_nifti(pair_dir + PRIOR_DEFORMED_MASK, deformed_sphere, affine, header)
    # save_nifti(pair_dir + PRIOR_DEFORMED_MASS, working_data, affine, header)

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
    working_data, mask = add_mass(prior, seg, prior_pos, radius, margin)
    apply_mask(prior, working_data, mask)
    return prior


def create_current_ct(current: np.ndarray, seg: np.ndarray,
                      current_pos: tuple[int, int, int], radius: int, margin: int,
                      affine: np.ndarray, header: nib.Nifti1Header, pair_dir: str):
    working_data, mask = add_mass(current, seg, current_pos, radius, margin)
    apply_mask(current, working_data, mask)
    return current


def rotate_and_drr(data: np.ndarray, angles: tuple[float, float, float]) -> np.ndarray:
    rotated_ct = rotate_ct_scan(data, angles[0], angles[1], angles[2])
    drr = create_drr_from_ct(rotated_ct)
    # save_drr(drr, output_filename)
    return drr


def create_heatmap(current_drr: np.ndarray, current_pp, prior_rotated_to_current_drr: np.ndarray,
                   heatmap_path: str) -> None:
    # Calculate the difference
    heatmap = np.asarray(current_drr) - np.asarray(prior_rotated_to_current_drr)

    # Create a figure with a single axis
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
    # bbox_inches='tight' removes extra white space around the image
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)

    # Close the plot to free up memory
    plt.close(fig)


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
    # load data
    data, affine, header = load_nifti(input_path)
    seg, _, _ = load_nifti(seg_path)

    margin = radius
    pair_dir = get_pair_dir(pair_index, input_path)

    # create prior ct
    prior_data = data.copy()
    prior_data = create_prior_ct(prior_data, seg,
                                 prior_pos, radius, margin,
                                 affine, header, pair_dir)

    # create current ct
    current_data = data
    current_data = create_current_ct(current_data, seg,
                                     current_pos, radius, margin,
                                     affine, header, pair_dir)

    # rotate ct files and create drr
    current_drr = rotate_and_drr(current_data, current_angles)
    prior_rotated_to_current_drr = rotate_and_drr(prior_data.copy(), current_angles)
    prior_rotated_to_prior_drr = rotate_and_drr(prior_data, prior_angles)

    # pp and save drr
    current_pp = apply_drr_post_processing(current_drr)
    save_drr(current_pp, pair_dir + CURRENT_FILENAME)

    prior_by_prior_pp = apply_drr_post_processing(prior_rotated_to_prior_drr)
    save_drr(prior_by_prior_pp, pair_dir + PRIOR_BY_PRIOR_FILENAME)

    prior_by_prior_pp = apply_drr_post_processing(prior_rotated_to_current_drr)
    save_drr(prior_by_prior_pp, pair_dir + PRIOR_BY_CURRENT_FILENAME)

    # create heatmap
    create_heatmap(current_drr,current_pp, prior_rotated_to_current_drr, pair_dir + HEATMAP_FILENAME)

    print("Done!")
