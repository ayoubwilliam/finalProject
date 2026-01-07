import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import sys
import os
import torch
import kornia
from torchvision.transforms.v2.functional import adjust_sharpness

from file_handler import load_nifti
from device_constants import DEVICE


##################### pre processing #####################

AIR_CT_THRESHOLD = -1000
CROP_MAX = 3000


def ct_pre_processing(ct_data):
    """Pre-process CT data using PyTorch"""
    # Convert to torch tensor and move to device
    # ct_tensor = torch.from_numpy(ct_data).float().to(DEVICE)
    ct_tensor = ct_data

    # Clip values
    ct_pre_processed = torch.clip(ct_tensor, AIR_CT_THRESHOLD, CROP_MAX)

    return ct_pre_processed


##################### post processing #####################

def normalize(image):
    """Normalize the image for display using PyTorch"""
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float().to(DEVICE)

    normalized = (image - image.min()) / (image.max() - image.min())
    return normalized


def flip(image):
    """Rotate image 90° to correct flipped orientation from CT coordinate system"""
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float().to(DEVICE)

    # Transpose and flip using PyTorch operations
    flipped = torch.flip(image.T, dims=[0])
    return flipped


def hist_equalize(image):
    """Equalize histogram equalization"""
    # Convert to numpy for skimage processing
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image

    equalized = exposure.equalize_hist(image_np)
    return torch.from_numpy(equalized).float().to(DEVICE)


def clahe(image, window_size=128, clip_limit=0.01):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
        image (torch.Tensor or np.ndarray): Input image (2D array).
        window_size (int or tuple, optional): window size
        clip_limit (float, optional): helps histogram look "More distributed"

    Returns:
        torch.Tensor: Equalized image as a float in the range [0.0, 1.0].
    """
    # Convert to torch tensor if needed
    if not isinstance(image, torch.Tensor):
        image_tensor = torch.from_numpy(image.copy()).float()
    else:
        image_tensor = image.clone()

    # Move to device and add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    image_tensor = image_tensor.to(DEVICE).unsqueeze(0).unsqueeze(0)

    # Apply CLAHE using kornia
    equalized_tensor = kornia.enhance.equalize_clahe(
        image_tensor,
        clip_limit=clip_limit,
        grid_size=(window_size, window_size)
    )

    # Remove extra dimensions but keep on device
    equalized_image = equalized_tensor.squeeze(0).squeeze(0)

    return equalized_image


def crop(image, y_start, y_end, x_start, x_end):
    """
    Manually crops an image based on specified pixel coordinates.

    Args:
        image (torch.Tensor or np.ndarray): The input image (2D array).
        y_start (int): Starting row index (top, inclusive).
        y_end (int): Ending row index (bottom, exclusive).
        x_start (int): Starting column index (left, inclusive).
        x_end (int): Ending column index (right, exclusive).

    Returns:
        torch.Tensor: The manually cropped image.
    """
    # Convert to torch tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float().to(DEVICE)

    # Basic bounds checking
    rows, cols = image.shape
    y_start = max(0, y_start)
    y_end = min(rows, y_end)
    x_start = max(0, x_start)
    x_end = min(cols, x_end)

    # Use PyTorch slicing
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image


def sharpen_image(image, sharpness_factor=1.0):
    """
    Apply sharpness adjustment to an image.

    Args:
        image (torch.Tensor or np.ndarray): Input image (2D array).
        sharpness_factor (float): Sharpness factor.
                                  0 = blurred image
                                  1 = original image
                                  >1 = sharpened image

    Returns:
        torch.Tensor: Sharpened image as a torch tensor.
    """
    # Convert to torch tensor if needed
    if not isinstance(image, torch.Tensor):
        image_tensor = torch.from_numpy(image.copy()).float()
    else:
        image_tensor = image.clone()

    # Move to device and add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    image_tensor = image_tensor.to(DEVICE).unsqueeze(0).unsqueeze(0)

    # Apply sharpness adjustment
    sharpened_tensor = adjust_sharpness(image_tensor, sharpness_factor)

    # Remove extra dimensions but keep on device
    sharpened_image = sharpened_tensor.squeeze(0).squeeze(0)

    return sharpened_image



from torchvision.transforms.functional import resize

def resize_image(image, target_size=(512, 512)):
    """Simple resize with stretching. Expects a Tensor."""
    # We unsqueeze(0) to add a channel dimension (H,W -> 1,H,W) for the function to work,
    # then squeeze(0) to remove it back to (H,W).
    return resize(image.unsqueeze(0), target_size, antialias=True).squeeze(0)


def apply_drr_post_processing(drr_xray, window_size=8, clip_limit=8.0, sharpness_factor=3.0):
    """Post-process DRR using PyTorch operations"""
    drr_xray = clahe(drr_xray, window_size, clip_limit)
    drr_xray = sharpen_image(drr_xray, sharpness_factor)
    return drr_xray


##################### main drr #####################
def create_drr_from_ct(ct_data: np.array, projection_axis: int = 1):
    # perform pre-processing
    ct_pre_processed = ct_pre_processing(ct_data)

    # creating drr- sum using PyTorch
    drr_image = torch.sum(ct_pre_processed, dim=projection_axis)

    # rotate 90 deg
    rotated_image = flip(drr_image)

    # normalize to 0-1
    normalized_image = normalize(rotated_image)

    # UPDATED: Call the simple resize function
    normalized_image = resize_image(normalized_image)

    return normalized_image


def save_drr(drr, output_path):
    print(f"saving to {output_path}")
    plt.imsave(output_path, drr.cpu().numpy(), cmap='gray')


def create_drr_with_processing(ct_data: np.array, projection_axis: int = 1,
                               window_size: int = 8, clip_limit: float = 8.0,
                               sharpness_factor: float = 3.0) -> np.ndarray:
    ct_pre_processed = ct_pre_processing(ct_data)

    # Sum using PyTorch
    drr_image = torch.sum(ct_pre_processed, dim=projection_axis)

    xray_post_processed = apply_drr_post_processing(drr_image, window_size, clip_limit, sharpness_factor)

    # Convert to numpy
    xray_np = xray_post_processed.cpu().numpy()

    return xray_np


def create_drr(ct_data: np.array, output_path: str, projection_axis: int = 1) -> np.ndarray:
    # Sum using PyTorch
    ct_tensor = torch.from_numpy(ct_data).float().to(DEVICE)

    # Creating drr
    drr_image = torch.sum(ct_tensor, dim=projection_axis)

    drr_image = normalize(drr_image)
    drr_image = flip(drr_image)

    # Convert to numpy for saving
    xray_np = drr_image.cpu().numpy()

    plt.imsave(output_path, xray_np, cmap='gray')

    return xray_np


if __name__ == "__main__":
    # Define input and output paths
    input_nifti_path = "../../ct/1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.nii.gz"
    data, affine, header = load_nifti(input_nifti_path)
    output_dir = "../drr_output/kornia"

    # Define the parameter ranges
    sharpness_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    window_sizes = [4, 8, 16]
    clip_limit = 8.0

    # # Define the parameter ranges
    # clip_limits = [1.0, 8.0, 16.0, 20.0]
    # window_sizes = [4, 8, 16]
    #
    # # Iterate through all combinations
    # for clip in clip_limits:
    #     for window in window_sizes:
    #         output_filename = f"kornia_clahe_clip{clip}_window{window}.png"
    #         output_path = os.path.join(output_dir, output_filename)
    #         print(f"Processing: clip_limit={clip}, window_size={window}")
    #         create_drr_from_ct(data, output_path, 1, window_size=window, clip_limit=clip

    # Iterate through all combinations
    for sharpness in sharpness_values:
        for window in window_sizes:
            output_filename = f"drr_clip{clip_limit}_window{window}_sharp{sharpness}.png"
            output_path = os.path.join(output_dir, output_filename)
            print(f"Processing: clip_limit={clip_limit}, window_size={window}, sharpness={sharpness}")
            create_drr_from_ct(data, output_path, 1,
                               window_size=window,
                               clip_limit=clip_limit,
                               sharpness_factor=sharpness)
