import sys
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import binary_dilation, ball

from file_handler import load_nifti, save_nifti
from create_shapes import create_box, create_sphere, create_ellipsoid

KERNEL_SIZE = 13
STRIDE = 2


def mask_to_ranges(mask: np.ndarray):
    """
    This function computes a tight axis-aligned bounding box for 3D boolean mask and returns its index ranges.
        parameters:
        1) mask: 3D boolean array where True marks the region of interest.
        returns: ranges: list [[x0, x1], [y0, y1], [z0, z1]] with exclusive upper bounds along each axis.
    """
    # Find all voxel coordinates where the mask is True
    coords = np.argwhere(mask)

    # Compute minimal and maximal indices for each axis
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1  # +1 so upper bound is exclusive

    # Return per-axis index ranges
    return [[x0, x1], [y0, y1], [z0, z1]]


def get_expanded_roi(mask: np.ndarray, kernel_size: int, shape: tuple[int, int, int]):
    """
    This function expands the mask bounding box by kernel_size and clips it to the full volume shape.
        parameters:
        1) mask: 3D boolean array used to define the original bounding box.
        2) kernel_size: padding size applied around the bounding box in all directions.
        3) shape: full volume shape used to clamp the expanded bounding box.
        returns: roi_bounds: tuple (x0, x1, y0, y1, z0, z1) defining the expanded region of interest.
    """
    # Get the tight bounding box around the mask
    (x0, x1), (y0, y1), (z0, z1) = mask_to_ranges(mask)

    # Expand bounds while keeping them inside the volume
    x0 = max(0, x0 - kernel_size)
    y0 = max(0, y0 - kernel_size)
    z0 = max(0, z0 - kernel_size)
    x1 = min(shape[0], x1 + kernel_size)
    y1 = min(shape[1], y1 + kernel_size)
    z1 = min(shape[2], z1 + kernel_size)

    # Return all six bounds as one tuple
    return x0, x1, y0, y1, z0, z1


def get_pooling(data: np.ndarray, mask: np.ndarray,
                kernel_size: int, stride: int,
                use_max_pooling: bool) -> np.ndarray:
    """
    This function applies 3D pooling on an expanded ROI around a mask and returns the pooled ROI.
       parameters:
       1) data: 3D volume containing the original values.
       2) mask: 3D boolean array that defines the region of interest.
       3) kernel_size: pooling kernel size used along each spatial dimension.
       4) stride: pooling stride used along each spatial dimension.
       5) use_max_pooling: flag that selects max pooling when True, average pooling otherwise.
       returns: pooled_region: 3D array holding the pooled values over the ROI with the same shape as the ROI.
   """
    # Compute the expanded ROI bounds based on the mask and kernel size
    x0, x1, y0, y1, z0, z1 = get_expanded_roi(mask, kernel_size, data.shape)

    # Extract the ROI slice from the full volume
    region = data[x0:x1, y0:y1, z0:z1]

    # Convert ROI to a torch tensor with shape [batch, channel, depth, height, width]
    x = torch.from_numpy(region).unsqueeze(0).unsqueeze(0).float()

    # Apply either max pooling or average pooling
    if use_max_pooling:
        pooled = F.max_pool3d(x, kernel_size, stride)
    else:
        pooled = F.avg_pool3d(x, kernel_size, stride)

    # Upsample pooled tensor back to the original ROI shape
    pooled_resized = F.interpolate(pooled, size=region.shape, mode="nearest")

    # Drop batch and channel dimensions and convert back to NumPy
    pooled_region = pooled_resized.squeeze(0).squeeze(0).cpu().numpy()

    # Return the pooled region with the same shape as the ROI
    return pooled_region


def apply_pooling(data: np.ndarray, mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    This function smooths values strictly inside the mask using pooled ROI values.
    Dilation has been removed.
    """
    # Compute ROI bounds to work on a smaller chunk of data
    x0, x1, y0, y1, z0, z1 = get_expanded_roi(mask, kernel_size, data.shape)

    # Compute pooled values for the ROI
    pooled = get_pooling(data, mask, kernel_size, STRIDE, False)

    # Cut the mask to the same ROI dimensions
    local_mask = mask[x0:x1, y0:y1, z0:z1]

    # Get a writable view into the ROI portion of the original data
    region_view = data[x0:x1, y0:y1, z0:z1]

    # Overwrite only voxels inside the local mask with the pooled values
    region_view[local_mask] = pooled[local_mask]

    # Return the full volume with the updated ROI
    return data


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data, affine, header = load_nifti(input_path)

    # mask = create_box(data, [[163, 203], [233, 273], [276, 316]])
    mask = create_sphere(data, [180, 250, 300], 30)
    # mask = create_ellipsoid(data, [180, 250, 300], [30, 40, 60])

    avg_pooling = apply_pooling(data, mask, KERNEL_SIZE)

    save_nifti(output_path, avg_pooling, affine, header)
