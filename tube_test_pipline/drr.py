"""
DRR (Digitally Reconstructed Radiograph) creation and post-processing.
Pipeline: CT pre-processing → sum projection → flip → normalize → resize → CLAHE → sharpen.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import torch
import kornia
from torchvision.transforms.v2.functional import adjust_sharpness
from torchvision.transforms.functional import resize

import config as cfg


# ======================== PRE-PROCESSING ========================

def ct_pre_processing(ct_data):
    """Clips CT values to [AIR_CT_THRESHOLD, CT_CLIP_MAX]."""
    ct_tensor = ct_data
    ct_pre_processed = torch.clip(ct_tensor, cfg.AIR_CT_THRESHOLD, cfg.CT_CLIP_MAX)
    return ct_pre_processed


# ======================== POST-PROCESSING ========================

def normalize(image):
    """Min-max normalizes an image to [0, 1]."""
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float().to(cfg.DEVICE)

    normalized = (image - image.min()) / (image.max() - image.min())
    return normalized


def flip(image):
    """Rotates image 90° to correct flipped orientation from CT coordinate system."""
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float().to(cfg.DEVICE)

    flipped = torch.flip(image.T, dims=[0])
    return flipped


def hist_equalize(image):
    """Applies histogram equalization via skimage."""
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image

    equalized = exposure.equalize_hist(image_np)
    return torch.from_numpy(equalized).float().to(cfg.DEVICE)


def clahe(image, window_size=None, clip_limit=None):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) via kornia."""
    if window_size is None:
        window_size = cfg.CLAHE_WINDOW_SIZE
    if clip_limit is None:
        clip_limit = cfg.CLAHE_CLIP_LIMIT

    if not isinstance(image, torch.Tensor):
        image_tensor = torch.from_numpy(image.copy()).float()
    else:
        image_tensor = image.clone()

    # (H, W) → (1, 1, H, W)
    image_tensor = image_tensor.to(cfg.DEVICE).unsqueeze(0).unsqueeze(0)

    equalized_tensor = kornia.enhance.equalize_clahe(
        image_tensor,
        clip_limit=clip_limit,
        grid_size=(window_size, window_size),
    )

    equalized_image = equalized_tensor.squeeze(0).squeeze(0)
    return equalized_image


def sharpen_image(image, sharpness_factor=None):
    """Applies sharpness adjustment. Factor > 1 sharpens, < 1 blurs."""
    if sharpness_factor is None:
        sharpness_factor = cfg.SHARPNESS_FACTOR

    if not isinstance(image, torch.Tensor):
        image_tensor = torch.from_numpy(image.copy()).float()
    else:
        image_tensor = image.clone()

    # (H, W) → (1, 1, H, W)
    image_tensor = image_tensor.to(cfg.DEVICE).unsqueeze(0).unsqueeze(0)

    sharpened_tensor = adjust_sharpness(image_tensor, sharpness_factor)

    sharpened_image = sharpened_tensor.squeeze(0).squeeze(0)
    return sharpened_image


def resize_image(image, target_size=None):
    """Resizes a 2D tensor to target_size with antialiasing."""
    if target_size is None:
        target_size = cfg.DRR_TARGET_SIZE
    return resize(image.unsqueeze(0), target_size, antialias=True).squeeze(0)


def apply_drr_post_processing(drr_xray, window_size=None, clip_limit=None, sharpness_factor=None):
    """Applies CLAHE then sharpening to a raw DRR image."""
    if window_size is None:
        window_size = cfg.CLAHE_WINDOW_SIZE
    if clip_limit is None:
        clip_limit = cfg.CLAHE_CLIP_LIMIT
    if sharpness_factor is None:
        sharpness_factor = cfg.SHARPNESS_FACTOR

    drr_xray = clahe(drr_xray, window_size, clip_limit)
    drr_xray = sharpen_image(drr_xray, sharpness_factor)
    return drr_xray


# ======================== MAIN DRR ========================

def create_drr_from_ct(ct_data, projection_axis=None):
    """Creates a DRR from a 3D CT volume: pre-process → sum → flip → normalize → resize."""
    if projection_axis is None:
        projection_axis = cfg.DRR_PROJECTION_AXIS

    ct_pre_processed = ct_pre_processing(ct_data)
    drr_image = torch.sum(ct_pre_processed, dim=projection_axis)
    rotated_image = flip(drr_image)
    normalized_image = normalize(rotated_image)
    normalized_image = resize_image(normalized_image)

    return normalized_image


def save_drr(drr, output_path):
    """Saves a DRR tensor as a grayscale PNG."""
    print(f"saving to {output_path}")
    plt.imsave(output_path, drr.cpu().numpy(), cmap='gray')
