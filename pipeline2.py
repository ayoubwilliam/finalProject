import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from bspline import get_deformed_sphere_fast
from file_handler import load_nifti, save_nifti
from drr_with_post_processing import create_drr_from_ct, save_drr
import matplotlib.pyplot as plt

from project.finalProject.drr_with_post_processing import apply_drr_post_processing
from project.finalProject.pooling import apply_pooling
from project.finalProject.rotation import rotate_ct_scan

colors = [
    (0, 1, 0, 1),  # Green (neg values)
    (0, 1, 0, 0),  # Transparent (Approaching 0 from positive)
    (1, 0, 0, 0),  # Transparent (Approaching 0 from negative)
    (1, 0, 0, 1)  # Red (pos values)
]
custom_cmap = LinearSegmentedColormap.from_list("RedClearGreen", colors, N=256)


def correct_mask_by_seg(mask, seg):
    return mask.astype(bool) & seg.astype(bool)


def apply_mask(destination_data, source_data, mask):
    destination_data[mask] = source_data[mask]


def pipeline():
    input_path = "../../ct/1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.nii.gz"
    seg_path = "../../ct/1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053_lungs_seg.nii.gz"
    intensity = 25
    radius = 30
    margin = radius
    grid_density_factor = 16
    deformation_factor = 0.2
    prior_pos = [180, 250, 250]
    #
    current_pos = [340, 300, 400]
    prior_angle = [20, 10, 0]
    current_angle = [-20, -10, 0]
    pooling_kernel_size = 8

    # todo create prior

    # Test fast method
    print("Running add_deformed_sphere_fast...")
    data, affine, header = load_nifti(input_path)
    seg, _ , _ = load_nifti(seg_path)

    prior = data.copy()
    deformed_sphere, mask = get_deformed_sphere_fast(prior.shape, intensity, prior_pos, radius, margin,
                                                     grid_density_factor, deformation_factor)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(prior, deformed_sphere, mask)
    save_nifti("./pipeline_output/prior_bspline_fast_mask.nii.gz", deformed_sphere, affine, header)
    save_nifti("./pipeline_output/prior_bspline_fast_mask.nii.gz", deformed_sphere, affine, header)

    print("applying pooling...")
    # prior,mask = apply_pooling(prior, mask, pooling_kernel_size)
    pooled_data, mask = apply_pooling(prior, mask, pooling_kernel_size)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(prior, pooled_data, mask)
    save_nifti("./pipeline_output/prior_pooling.nii.gz", prior, affine, header)
    print("finished pooling.")

    # todo create current
    current = data
    deformed_sphere, mask = get_deformed_sphere_fast(current.shape, intensity, current_pos, radius, margin,
                                                     grid_density_factor, deformation_factor)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(current, deformed_sphere, mask)

    save_nifti("./pipeline_output/post_bspline_fast_mask.nii.gz", deformed_sphere, affine, header)
    save_nifti("./pipeline_output/post_bspline_fast.nii.gz", current, affine, header)

    print("applying pooling...")
    # current,mask = apply_pooling(current, mask, pooling_kernel_size)
    pooled_data, mask = apply_pooling(current, mask, pooling_kernel_size)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(current, pooled_data, mask)
    save_nifti("./pipeline_output/current_pooling.nii.gz", current, affine, header)
    print("finished pooling.")

    rotated_current = rotate_ct_scan(current, current_angle[0], current_angle[1], current_angle[2])
    current_drr = create_drr_from_ct(rotated_current)
    save_drr(current_drr, "./pipeline_output/current.png")

    # todo prior rotated to current
    prior_rotated_to_current = rotate_ct_scan(prior.copy(), current_angle[0], current_angle[1], current_angle[2])
    prior_rotated_to_current_drr = create_drr_from_ct(prior_rotated_to_current)
    save_drr(prior_rotated_to_current_drr, "./pipeline_output/prior_rotated_to_current.png")

    # todo prior rotated to prior
    prior_rotated_to_prior = rotate_ct_scan(prior, prior_angle[0], prior_angle[1], prior_angle[2])
    prior_rotated_to_prior_drr = create_drr_from_ct(prior_rotated_to_prior, )
    save_drr(prior_rotated_to_prior_drr, "./pipeline_output/prior_rotated_to_prior.png")

    # todo create heatmap
    heatmap = np.asarray(current_drr) - np.asarray(prior_rotated_to_current_drr)

    # Apply post-processing to both images for consistent display
    current_pp = apply_drr_post_processing(current_drr)
    prior_rotated_to_prior_pp = apply_drr_post_processing(prior_rotated_to_prior_drr)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Prior (Rotated to its original Prior angle)
    axes[0].imshow(prior_rotated_to_prior_pp, cmap='gray')
    axes[0].set_title("Prior (Rotated to Prior Angle)")
    axes[0].axis('off')

    # Plot 2: Current (Target image)
    axes[1].imshow(current_pp, cmap='gray')
    axes[1].set_title("Current Image")
    axes[1].axis('off')

    # Plot 3: Heatmap Overlay (Difference on top of Current)
    axes[2].imshow(current_pp, cmap='gray')  # Background

    max_error = np.max(np.abs(heatmap))
    im = axes[2].imshow(heatmap, cmap=custom_cmap, alpha=1, vmin=-max_error, vmax=max_error)  # Overlay
    axes[2].set_title("Difference Heatmap")
    axes[2].axis('off')

    # Add a colorbar for the heatmap
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    plt.tight_layout()
    plt.show()

    print("Done!")


if __name__ == "__main__":
    pipeline()
