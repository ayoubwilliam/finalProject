"""
Module: preprocessing.py
Provides functionality for preprocessing.
"""

import torch
import numpy as np
from PIL import Image
import kornia
from torchvision.transforms.v2.functional import adjust_sharpness
from torchvision.transforms.functional import resize

# Import config from the same root directory
import config as cfg


# ==========================================
# CROP
# 1. Trim a fixed pixel margin off each edge (drops border markers/artifacts).
# 2. Center crop to a fraction of the remaining height/width.
# 3. Force square by taking the min of the two resulting dimensions,
#    center-cropped from whichever side is larger.
# ==========================================

def crop_to_square(image: torch.Tensor, edge_trim_px: int, center_frac: float, dy: float = 0.0, dx: float = 0.0):
    """Crops a 2D X-ray tensor to a centered square, trimming edges first.
       Returns the cropped image and its bounding box (y1, y2, x1, x2) relative to original image.
       dy and dx are floats between -1.0 (top/left) and 1.0 (bottom/right)."""
    h, w = image.shape

    # 1. Trim fixed edge margin
    trimmed_start_y = edge_trim_px
    trimmed_start_x = edge_trim_px
    trimmed = image[edge_trim_px:h - edge_trim_px, edge_trim_px:w - edge_trim_px]
    th, tw = trimmed.shape

    # 2. Determine square size based on center_frac
    crop_h = int(th * center_frac)
    crop_w = int(tw * center_frac)
    square_size = min(crop_h, crop_w)

    # 3. Calculate translation shifts based on dy, dx within the trimmed region
    max_shift_y = th - square_size
    max_shift_x = tw - square_size

    # Map [-1.0, 1.0] to [0, max_shift]
    shift_y = int((dy + 1.0) / 2.0 * max_shift_y)
    shift_x = int((dx + 1.0) / 2.0 * max_shift_x)

    # Calculate absolute coordinates
    final_y1 = trimmed_start_y + shift_y
    final_x1 = trimmed_start_x + shift_x
    final_y2 = final_y1 + square_size
    final_x2 = final_x1 + square_size

    return image[final_y1:final_y2, final_x1:final_x2], (final_y1, final_y2, final_x1, final_x2)


# ==========================================
# EXISTING PREPROCESSING STEPS (unchanged)
# ==========================================

def normalize(image):
    """Min-max normalizes an image to [0, 1]."""
    normalized = (image - image.min()) / (image.max() - image.min())
    return normalized


def clahe(image, window_size=None, clip_limit=None):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) via kornia."""
    if window_size is None:
        window_size = cfg.CLAHE_WINDOW_SIZE
    if clip_limit is None:
        clip_limit = cfg.CLAHE_CLIP_LIMIT

    # (H, W) → (1, 1, H, W)
    image_tensor = image.unsqueeze(0).unsqueeze(0)

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

    # (H, W) → (1, 1, H, W)
    image_tensor = image.unsqueeze(0).unsqueeze(0)

    sharpened_tensor = adjust_sharpness(image_tensor, sharpness_factor)

    sharpened_image = sharpened_tensor.squeeze(0).squeeze(0)
    return sharpened_image


def resize_image(image, target_size=None):
    """Resizes a 2D tensor to target_size with antialiasing."""
    if target_size is None:
        target_size = cfg.DRR_TARGET_SIZE
    return resize(image.unsqueeze(0), target_size, antialias=True).squeeze(0)


def preprocess_real_image(img_path, edge_trim_px: int, center_frac: float, dy: float = 0.0, dx: float = 0.0):
    """
    Loads a real JPG X-ray and applies the DRR post-processing pipeline,
    now with a parameterized square crop and translation around the body/lung region before resizing.
    Returns the processed tensor, numpy array, and the crop bounding box (y1, y2, x1, x2).
    """
    # 1. Load image and convert to PyTorch tensor on the active device
    img = Image.open(img_path).convert('L')
    img_np = np.array(img, dtype=np.float32)
    img_tensor = torch.from_numpy(img_np).to(cfg.DEVICE)

    # 2. Crop to a square region with the given settings
    img_tensor, bbox = crop_to_square(img_tensor, edge_trim_px, center_frac, dy, dx)
    
    img_tensor = resize_image(img_tensor)
    img_tensor = normalize(img_tensor)
    img_tensor = clahe(img_tensor)
    img_tensor = sharpen_image(img_tensor)

    # 3. Format dimensions for the model: [Batch, Channel, H, W]
    final_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    return final_tensor, img_tensor.cpu().numpy(), bbox