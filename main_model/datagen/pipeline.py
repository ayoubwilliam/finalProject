"""
Core data generation pipeline for a single prior-current pair.
Pipeline per pair: [add_mass] → [add_tube] → rotate → crop → DRR → post-process → heatmap.

Supports both deformed ball masses and intubation tubes, controlled by
cfg.ADD_CONSOLIDATION and cfg.ADD_TUBE toggles.

GPU memory is carefully managed: current CT is fully processed and deleted before
prior CT is loaded, ensuring only one full-size volume is in VRAM at a time.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import config as cfg
from lib.overlay_heatmap import build_overlay_colormap
from lib.nifti_io import save_image_as_nifti
from datagen.deformation import get_deformed_sphere_fast
from datagen.pooling import apply_pooling
from datagen.rotation import rotate_ct_scan
from datagen.drr import create_drr_from_ct, save_drr, apply_drr_post_processing
from datagen.crop import get_segmentation_bounds
from datagen.tube import add_tube_to_ct

custom_cmap = build_overlay_colormap()


def correct_mask_by_seg(mask, seg):
    """Intersects the deformed mass mask with the lung segmentation."""
    return mask.bool() & seg.bool()


def apply_mask(destination_data, source_data, mask):
    """Copies source into destination only where source > destination and mask is True."""
    positive_mask = (destination_data < source_data) & mask
    destination_data[positive_mask] = source_data[positive_mask]


def add_mass(data, seg, pos, radius, margin):
    """Generates a deformed sphere, smooths its boundary via pooling, and applies it to the volume."""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float().to(cfg.DEVICE)
    if isinstance(seg, np.ndarray):
        seg = torch.from_numpy(seg).bool().to(cfg.DEVICE)

    working_data = data.clone()
    print("Running add_deformed_sphere_fast...")

    deformed_sphere, mask = get_deformed_sphere_fast(
        working_data, cfg.MASS_INTENSITY, pos, radius, margin,
        cfg.GRID_DENSITY_FACTOR, cfg.DEFORMATION_FACTOR,
    )

    mask = correct_mask_by_seg(mask, seg)
    apply_mask(working_data, deformed_sphere, mask)
    print("finished deformed_sphere_fast...")

    # boundary smoothing via pooling
    print("start pooling...")
    pooled_data, mask = apply_pooling(working_data, mask, cfg.POOLING_KERNEL_SIZE)
    mask = correct_mask_by_seg(mask, seg)
    apply_mask(working_data, pooled_data, mask)
    print("finished pooling.")

    return working_data, mask


def create_prior_ct(prior, seg, prior_pos, radius, margin):
    """Adds a deformed mass to the prior CT volume."""
    working_data, mask = add_mass(prior, seg, prior_pos, radius, margin)
    apply_mask(prior, working_data, mask)
    return prior


def create_current_ct(current, seg, current_pos, radius, margin):
    """Adds a deformed mass to the current CT volume."""
    working_data, mask = add_mass(current, seg, current_pos, radius, margin)
    apply_mask(current, working_data, mask)
    return current


def rotate_and_drr(data, angles, seg, crop_margin=None, use_crop=True):
    """Rotates a 3D CT, crops to segmentation bounds (optional), and generates a 2D DRR."""
    if crop_margin is None:
        crop_margin = cfg.CROP_MARGIN

    rotated_ct_gpu = rotate_ct_scan(data, angles[0], angles[1], angles[2])

    # move to CPU immediately to free GPU for cropping
    rotated_ct_cpu = rotated_ct_gpu.detach().cpu()
    del rotated_ct_gpu
    torch.cuda.empty_cache()

    if use_crop:
        bounds = get_segmentation_bounds(seg, angles, crop_margin)
        torch.cuda.empty_cache()

        if bounds is None:
            print("Warning: Empty segmentation. Using full rotated volume.")
            cropped_ct_gpu = rotated_ct_cpu.to(cfg.DEVICE)
        else:
            z1, z2, y1, y2, x1, x2 = bounds
            # crop on CPU first, then reload only the cropped portion to GPU
            cropped_ct_cpu = rotated_ct_cpu[z1:z2, y1:y2, x1:x2]
            cropped_ct_gpu = cropped_ct_cpu.to(cfg.DEVICE)
    else:
        torch.cuda.empty_cache()
        cropped_ct_gpu = rotated_ct_cpu.to(cfg.DEVICE)

    del rotated_ct_cpu

    drr = create_drr_from_ct(cropped_ct_gpu)
    return drr


def create_heatmap(current_drr, current_pp, prior_rotated_to_current_drr, heatmap_path):
    """Computes difference heatmap and saves both overlay PNG and raw NIfTI."""

    def ensure_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    current_drr = ensure_numpy(current_drr)
    prior_rotated_to_current_drr = ensure_numpy(prior_rotated_to_current_drr)
    current_pp = ensure_numpy(current_pp)

    heatmap = current_drr - prior_rotated_to_current_drr
    # heatmap[np.abs(heatmap) < cfg.GT_THRESHOLD] = 0 #todo removed this while making the tube make sure it is needed/not
    max_error = np.max(np.abs(heatmap))

    base, ext = os.path.splitext(heatmap_path)
    overlay_path = f"{base}_overlay{ext}"

    # save overlay with background
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

    # save raw heatmap as NIfTI
    save_image_as_nifti(heatmap, base + ".nii.gz")


@torch.no_grad()
def pipeline(pair_dir, ct_data, lungs_mask, trachea_mask, radius,
             prior_pos, current_pos, prior_angles, current_angles,
             has_prior_mass=True, has_current_mass=True,
             has_prior_tube=True, has_current_tube=True,
             tube_diameter=8.0, tube_thickness=2,
             use_crop=True):
    """
    Full pipeline for one prior-current pair.
    Processes current CT first, frees GPU, then processes prior CT.

    Parameters
    ----------
    pair_dir : str
        Output directory for this pair.
    ct_data : numpy.ndarray
        Raw CT volume data.
    lungs_mask : numpy.ndarray
        Lung segmentation mask (used for crop + deformed ball placement).
    trachea_mask : numpy.ndarray or None
        Trachea segmentation mask (used for tube insertion only).
        Can be None if cfg.ADD_TUBE is False.
    radius : int
        Radius for the deformed ball.
    prior_pos, current_pos : tuple
        3D voxel positions for ball placement in prior/current.
    prior_angles, current_angles : tuple
        (angle_x, angle_y, angle_z) rotation angles.
    has_prior_mass, has_current_mass : bool
        Whether to add a deformed ball to the prior/current CT.
    has_prior_tube, has_current_tube : bool
        Whether to add an intubation tube to the prior/current CT.
    tube_diameter : float
        Diameter for the tubes in this pair.
    tube_thickness : int
        Thickness of the tubes in this pair.
    use_crop : bool
        Whether to crop based on lung segmentation bounds.
    """
    # move to GPU — 'data' is our read-only template
    data = torch.from_numpy(ct_data).float().to(cfg.DEVICE)
    seg = torch.from_numpy(lungs_mask).bool().to(cfg.DEVICE)

    # Determine cropping segmentation based on user requirements
    if cfg.ADD_CONSOLIDATION:
        print("Consolidation is active, cropping to lungs mask.")
        crop_seg = seg
    elif trachea_mask is not None:
        print("Only tube is active, cropping to trachea mask.")
        crop_seg = torch.from_numpy(trachea_mask).bool().to(cfg.DEVICE)
    else:
        print("Warning: Trachea mask is missing, falling back to lungs mask for cropping.")
        crop_seg = seg

    margin = radius

    # ==========================================
    # PHASE 1: Process Current CT
    # ==========================================
    print("current: ")
    current_data = data.clone()

    # --- Deformed mass (optional) ---
    if cfg.ADD_CONSOLIDATION and has_current_mass:
        current_data = create_current_ct(current_data, seg, current_pos, radius, margin)

    # --- Intubation tube (optional) ---
    # Track the actual diameter used (may shrink due to walk failures)
    actual_tube_diameter = tube_diameter
    actual_tube_thickness = tube_thickness
    if cfg.ADD_TUBE and has_current_tube and trachea_mask is not None:
        current_data, actual_tube_diameter, actual_tube_thickness = add_tube_to_ct(
            current_data, trachea_mask, tube_diameter, tube_thickness
        )

    current_drr = rotate_and_drr(current_data, current_angles, crop_seg, use_crop=use_crop)

    # CRITICAL: free VRAM before processing prior
    del current_data
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE 2: Process Prior CT
    # ==========================================
    print("prior: ")
    prior_data = data.clone()

    # --- Deformed mass (optional) ---
    if cfg.ADD_CONSOLIDATION and has_prior_mass:
        prior_data = create_prior_ct(prior_data, seg, prior_pos, radius, margin)

    # --- Intubation tube (optional) ---
    # Use the actual diameter from the current tube so both tubes have the same width
    if cfg.ADD_TUBE and has_prior_tube and trachea_mask is not None:
        prior_data, _, _ = add_tube_to_ct(
            prior_data, trachea_mask, actual_tube_diameter, actual_tube_thickness
        )

    # two DRRs from same prior volume — no extra clone needed
    prior_rotated_to_current_drr = rotate_and_drr(prior_data, current_angles, crop_seg, use_crop=use_crop)
    prior_rotated_to_prior_drr = rotate_and_drr(prior_data, prior_angles, crop_seg, use_crop=use_crop)

    # CRITICAL: free all large GPU tensors
    del prior_data
    del data
    del seg
    del crop_seg
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE 3: Post-Processing (CPU / Lightweight)
    # ==========================================
    print("post processing and drr...")

    current_pp = apply_drr_post_processing(current_drr)
    save_drr(current_pp, pair_dir + cfg.CURRENT_FILENAME)

    prior_by_prior_pp = apply_drr_post_processing(prior_rotated_to_prior_drr)
    save_image_as_nifti(current_pp.cpu().numpy(), pair_dir + "current.nii.gz")
    save_drr(prior_by_prior_pp, pair_dir + cfg.PRIOR_BY_PRIOR_FILENAME)

    prior_by_current_pp = apply_drr_post_processing(prior_rotated_to_current_drr)
    save_image_as_nifti(prior_by_prior_pp.cpu().numpy(), pair_dir + "prior.nii.gz")
    save_drr(prior_by_current_pp, pair_dir + cfg.PRIOR_BY_CURRENT_FILENAME)

    # create heatmap
    print("heatmap...")
    create_heatmap(current_drr, current_pp, prior_rotated_to_current_drr, pair_dir + cfg.HEATMAP_FILENAME)

    # Force free all remaining GPU memory before the next pair
    del current_drr, current_pp, prior_rotated_to_current_drr
    del prior_rotated_to_prior_drr, prior_by_prior_pp, prior_by_current_pp
    torch.cuda.empty_cache()

    print("Done!")
