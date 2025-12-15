import sys
import numpy as np
from skimage.morphology import dilation, erosion, closing, opening, ball

from file_handler import load_nifti, save_nifti
from create_shapes import create_box, create_sphere, create_ellipsoid

RADIUS = 6
BACKGROUND_INTENSITY = -800


def mask_to_ranges(mask: np.ndarray):
    """Convert a 3D boolean mask into coordinate bounding box ranges."""
    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    return [[x0, x1], [y0, y1], [z0, z1]]


def apply_dilation(data: np.ndarray, radius: int, intensity: int, mask: np.ndarray, ) -> np.ndarray:
    """
    Dilate the mask (3D) and paint the dilated region with the given intensity.
    """
    # Ensure binary mask
    binary_mask = mask > 0

    # Dilate the binary mask
    dilated_mask = dilation(binary_mask, ball(radius))

    # Write intensity into dilated mask region
    data[dilated_mask] = intensity
    return data


def apply_erosion(data: np.ndarray, radius: int, intensity: int, mask: np.ndarray) -> np.ndarray:
    """
    Erode the mask (3D) and paint the eroded region with the given intensity.
    """
    # Ensure binary mask
    binary_mask = mask > 0

    # Erode the binary mask
    eroded_mask = erosion(binary_mask, ball(radius))

    # Remove the original shape completely
    data[binary_mask] = BACKGROUND_INTENSITY

    # Paint only the eroded (smaller) shape
    data[eroded_mask] = intensity

    return data


def apply_manual_closing(data: np.ndarray, radius: int, intensity: int, mask: np.ndarray) -> np.ndarray:
    binary_mask = mask > 0
    footprint = ball(radius)

    dilated = dilation(binary_mask, footprint)
    closed = erosion(dilated, footprint)

    data[binary_mask] = BACKGROUND_INTENSITY
    data[closed] = intensity

    return data


def apply_closing(data: np.ndarray, radius: int, intensity: int, mask: np.ndarray) -> np.ndarray:
    binary_mask = mask > 0

    # Close the binary mask
    closed = closing(binary_mask, ball(radius))

    # Write intensity into closed mask region
    data[closed] = intensity
    return data


def apply_opening(data: np.ndarray, radius: int, intensity: int, mask: np.ndarray) -> np.ndarray:
    binary_mask = mask > 0

    # Open the binary mask
    opened_mask = opening(binary_mask, ball(radius))

    # Remove the original shape completely
    data[binary_mask] = BACKGROUND_INTENSITY

    # Write intensity into opened mask region
    data[opened_mask] = intensity
    return data


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """
    Combine multiple binary masks into a single union mask.
    """
    if not masks:
        raise ValueError("combine_masks requires at least one mask")

    combined = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        combined |= (m > 0)

    return combined


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    intensity = int(sys.argv[3])

    data, affine, header = load_nifti(input_path)

    # Choose shape (returns intensity mask)
    # dilation/erosion:
    # mask = create_box(data, [[163, 203], [233, 273], [276, 316]])
    # closing:
    # mask1 = create_sphere(data, [145, 250, 300], 23)
    # mask2 = create_sphere(data, [195, 250, 300], 23)
    # mask = combine_masks(mask1, mask2)
    # opening:
    mask1 = create_sphere(data, [180, 250, 300], 15)
    mask2 = create_sphere(data, [180, 250, 325], 5)
    mask = combine_masks(mask1, mask2)

    # Apply dilation
    # data = apply_dilation(data, RADIUS, intensity, mask)
    # data = apply_dilation(data, RADIUS, intensity, mask1)
    # data = apply_erosion(data, RADIUS, intensity, mask1)
    # data = apply_closing(data, RADIUS, intensity, mask)
    data = apply_opening(data, RADIUS, intensity, mask)

    save_nifti(output_path, data, affine, header)
