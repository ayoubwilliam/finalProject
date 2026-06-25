"""
Core data generation pipeline for a single prior-current pair.
Pipeline per pair: add_tube → rotate → crop → DRR → post-process → heatmap.

GPU memory is carefully managed: current CT is fully processed and deleted before
prior CT is loaded, ensuring only one full-size volume is in VRAM at a time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import config as cfg
from overlay_heatmap import build_overlay_colormap
from nifti_io import save_image_as_nifti
from basic_tube import build_tube_volume
from rotation import rotate_ct_scan
from drr import create_drr_from_ct, save_drr, apply_drr_post_processing
from crop import get_segmentation_bounds

custom_cmap = build_overlay_colormap()


def apply_mask(destination_data, source_data, mask):
    """Copies source into destination only where source > destination and mask is True."""
    positive_mask = (destination_data < source_data) & mask
    destination_data[positive_mask] = source_data[positive_mask]


def add_prebuilt_tube(data, hollow_volume):
    """Applies a prebuilt hollow tube to the volume."""
    hollow_volume_gpu = torch.from_numpy(hollow_volume).to(cfg.DEVICE)
    
    # We apply the tube directly using the intensity values inside hollow_volume
    mask = hollow_volume_gpu > 0
    apply_mask(data, hollow_volume_gpu, mask)
    
    del hollow_volume_gpu
    torch.cuda.empty_cache()
    
    return data


def rotate_and_drr(data, angles, seg, crop_margin=None, use_crop=True):
    """Rotates a 3D CT, crops to lung bounds (optional), and generates a 2D DRR."""
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
    heatmap[np.abs(heatmap) < cfg.GT_THRESHOLD] = 0
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


def pipeline(pair_dir, ct_data, lungs_mask,
             prior_tube_params, current_tube_params, prior_angles, current_angles,
             has_prior_tube=True, has_current_tube=True, use_crop=True):
    """
    Full pipeline for one prior-current pair.
    Pre-builds tubes first to ensure CuPy has maximum VRAM available, 
    then processes current CT, frees GPU, then processes prior CT.
    """
    mask_numpy = lungs_mask.astype(np.uint8)

    # PRE-BUILD TUBES (CPU/CuPy) BEFORE allocating heavy base templates in PyTorch
    print("Pre-building tubes to conserve VRAM...")
    if has_current_tube:
        current_hollow = build_tube_volume(ct_data.shape, mask_numpy, current_tube_params)
    else:
        current_hollow = None
        
    if has_prior_tube:
        prior_hollow = build_tube_volume(ct_data.shape, mask_numpy, prior_tube_params)
    else:
        prior_hollow = None

    # NOW allocate the heavy base templates
    data = torch.from_numpy(ct_data).float().to(cfg.DEVICE)
    seg = torch.from_numpy(lungs_mask).bool().to(cfg.DEVICE)
    
    # ==========================================
    # PHASE 1: Process Current CT
    # ==========================================
    print("current: ")
    current_data = data.clone()
    if has_current_tube and current_hollow is not None:
        current_data = add_prebuilt_tube(current_data, current_hollow)
        del current_hollow # Free CPU RAM early

    current_drr = rotate_and_drr(current_data, current_angles, seg, use_crop=use_crop)

    # CRITICAL: free VRAM before processing prior
    del current_data
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE 2: Process Prior CT
    # ==========================================
    print("prior: ")
    prior_data = data.clone()
    if has_prior_tube and prior_hollow is not None:
        prior_data = add_prebuilt_tube(prior_data, prior_hollow)
        del prior_hollow # Free CPU RAM early

    # two DRRs from same prior volume — no extra clone needed
    prior_rotated_to_current_drr = rotate_and_drr(prior_data, current_angles, seg, use_crop=use_crop)
    prior_rotated_to_prior_drr = rotate_and_drr(prior_data, prior_angles, seg, use_crop=use_crop)

    # CRITICAL: free all large GPU tensors
    del prior_data
    del data
    del seg
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE 3: Post-Processing (CPU / Lightweight)
    # ==========================================
    print("post processing and drr...")
    
    if not pair_dir.endswith(os.sep):
        pair_dir += os.sep
    os.makedirs(pair_dir, exist_ok=True)

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

    print("Done!")
