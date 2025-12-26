import sys
import numpy as np
import torch
import torch.nn.functional as F
# Re-added binary_erosion to create the inner boundary
from skimage.morphology import binary_dilation, binary_erosion, ball

from file_handler import load_nifti, save_nifti
from create_shapes import create_box, create_sphere, create_ellipsoid

KERNEL_SIZE = 13
STRIDE = 2


def mask_to_ranges(mask: np.ndarray):
    """
    This function computes a tight axis-aligned bounding box for 3D boolean mask and returns its index ranges.
    """
    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    return [[x0, x1], [y0, y1], [z0, z1]]


def get_expanded_roi(mask: np.ndarray, kernel_size: int, shape: tuple[int, int, int]):
    """
    This function expands the mask bounding box by kernel_size and clips it to the full volume shape.
    """
    (x0, x1), (y0, y1), (z0, z1) = mask_to_ranges(mask)

    x0 = max(0, x0 - kernel_size)
    y0 = max(0, y0 - kernel_size)
    z0 = max(0, z0 - kernel_size)
    x1 = min(shape[0], x1 + kernel_size)
    y1 = min(shape[1], y1 + kernel_size)
    z1 = min(shape[2], z1 + kernel_size)

    return x0, x1, y0, y1, z0, z1


def get_pooling(data: np.ndarray, mask: np.ndarray,
                kernel_size: int, stride: int,
                use_max_pooling: bool) -> np.ndarray:
    """
    This function applies 3D pooling on an expanded ROI around a mask and returns the pooled ROI.
    """
    x0, x1, y0, y1, z0, z1 = get_expanded_roi(mask, kernel_size, data.shape)
    region = data[x0:x1, y0:y1, z0:z1]

    x = torch.from_numpy(region).unsqueeze(0).unsqueeze(0).float()

    if use_max_pooling:
        pooled = F.max_pool3d(x, kernel_size, stride)
    else:
        pooled = F.avg_pool3d(x, kernel_size, stride)

    pooled_resized = F.interpolate(pooled, size=region.shape, mode="nearest")
    pooled_region = pooled_resized.squeeze(0).squeeze(0).cpu().numpy()

    return pooled_region


# todo change code style
def apply_pooling(data: np.ndarray, mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Optimized: smooths ONLY the boundary shell of the object using ROI cropping.
    1. Creates a 'shell' by subtracting the eroded mask from the dilated mask.
    2. Applies pooling only to this shell region.
    """
    # 1. Get the ROI bounds (expanded to fit the dilation/kernel)

    data = data.copy()
    x0, x1, y0, y1, z0, z1 = get_expanded_roi(mask, kernel_size, data.shape)

    # 2. Extract the local view of the mask
    #    (Working on this small slice is 100x faster than the full volume)
    local_mask = mask[x0:x1, y0:y1, z0:z1]

    # 3. Create the "Shell" Mask
    #    Dilation expands outwards, Erosion shrinks inwards.
    #    The XOR (^) gives us the pixels that are in one but not the other (the edge).
    structuring_element = ball(kernel_size)
    dilated = binary_dilation(local_mask, structuring_element)
    eroded = binary_erosion(local_mask, structuring_element)

    boundary_shell_mask = dilated ^ eroded

    # 4. Get the pooled values (calculated from original data)
    pooled = get_pooling(data, mask, kernel_size, STRIDE, False)

    # 5. Update data ONLY at the boundary shell
    region_view = data[x0:x1, y0:y1, z0:z1]

    # We only overwrite the 'boundary_shell_mask' pixels.
    # The center of the object (inside 'eroded') remains untouched/sharp.
    region_view[boundary_shell_mask] = pooled[boundary_shell_mask]

    # --- NEW STEP: Reconstruct the full-size mask ---
    # Create an empty volume with the same shape as the original data
    full_dilated_mask = np.zeros(data.shape, dtype=bool)

    # Paste the local dilated ROI back into the correct coordinates
    full_dilated_mask[x0:x1, y0:y1, z0:z1] = dilated

    return data, full_dilated_mask


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data, affine, header = load_nifti(input_path)

    # mask = create_box(data, [[163, 203], [233, 273], [276, 316]])
    mask = create_sphere(data, [180, 250, 300], 30)
    # mask = create_ellipsoid(data, [180, 250, 300], [30, 40, 60])

    avg_pooling = apply_pooling(data, mask, KERNEL_SIZE)

    save_nifti(output_path, avg_pooling, affine, header)
