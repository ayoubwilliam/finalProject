"""
Segmentation-based bounding box cropping.
Rotates the lung segmentation mask, then computes tight Y bounds while keeping X and Z perfectly square.
"""

import torch

import config as cfg
from datagen.rotation import rotate_ct_scan


# ==========================================
# 1. TENSOR & GPU HELPERS
# ==========================================

def _prepare_and_rotate_segmentation(segmentation, angles) -> torch.Tensor:
    """Handles GPU transfer, precision conversion, and rotation."""
    if not isinstance(segmentation, torch.Tensor):
        seg_tensor = torch.from_numpy(segmentation).to(cfg.DEVICE)
    else:
        seg_tensor = segmentation.to(cfg.DEVICE)

    seg_float = seg_tensor.half()
    rotated_seg = rotate_ct_scan(seg_float, angles[0], angles[1], angles[2])

    # Return as boolean mask
    return (rotated_seg > 0.5)


def _get_active_bounds(rotated_seg: torch.Tensor):
    """Finds the min and max coordinates of the lung mask in 3D space."""
    non_zero_indices = torch.nonzero(rotated_seg)
    if non_zero_indices.numel() == 0:
        return None, None

    min_vals = non_zero_indices.min(dim=0).values
    max_vals = non_zero_indices.max(dim=0).values
    return min_vals, max_vals


# ==========================================
# 2. GEOMETRY & MATH HELPERS
# ==========================================

def _shift_sliding_window(start: int, end: int, limit: int) -> tuple[int, int]:
    """
    Ensures a 1D bounding box stays within the array limits.
    If it goes out of bounds, it slides the entire box back inside without resizing it.
    """
    if start < 0:
        end -= start
        start = 0
    elif end > limit:
        start -= (end - limit)
        end = limit

    return start, end


def _calculate_square_crop(min1: int, max1: int, limit1: int,
                           min2: int, max2: int, limit2: int, margin: int) -> tuple:
    """
    Takes two dimensions and forces them into a perfect square around their centers,
    respecting physical array limits and margins.
    """
    # 1. Apply margins
    start1, end1 = min1 - margin, max1 + margin
    start2, end2 = min2 - margin, max2 + margin

    box1_size = end1 - start1
    box2_size = end2 - start2

    # 2. Find target size (Max of the two, capped by physical limits)
    target_size = max(box1_size, box2_size)
    target_size = min(target_size, limit1, limit2)

    # 3. Find centers
    center1 = start1 + box1_size // 2
    center2 = start2 + box2_size // 2

    # 4. Build new raw square bounds
    sq_start1 = center1 - target_size // 2
    sq_end1 = sq_start1 + target_size

    sq_start2 = center2 - target_size // 2
    sq_end2 = sq_start2 + target_size

    # 5. Slide windows if they hit the edges
    final_start1, final_end1 = _shift_sliding_window(sq_start1, sq_end1, limit1)
    final_start2, final_end2 = _shift_sliding_window(sq_start2, sq_end2, limit2)

    return final_start1, final_end1, final_start2, final_end2


# ==========================================
# 3. MAIN ORCHESTRATOR
# ==========================================

def get_segmentation_bounds(segmentation, angles, margin):
    """
    Forces Dim 0 (Z) and Dim 2 (X) into a perfect square in 3D.
    Tightly crops Dim 1 (Y).
    """
    with torch.no_grad():
        # 1. Rotate the mask
        rotated_seg = _prepare_and_rotate_segmentation(segmentation, angles)

        # 2. Find the edges of the lungs
        min_vals, max_vals = _get_active_bounds(rotated_seg)
        if min_vals is None:
            return None

        # Extract limits and coordinates
        dim0_limit, dim1_limit, dim2_limit = rotated_seg.shape
        dim0_min, dim1_min, dim2_min = min_vals[0].item(), min_vals[1].item(), min_vals[2].item()
        dim0_max, dim1_max, dim2_max = max_vals[0].item(), max_vals[1].item(), max_vals[2].item()

        # --- DIM 1 (Y - Front to Back) : Tight Crop ---
        crop_y1 = max(0, dim1_min - margin)
        crop_y2 = min(dim1_limit, dim1_max + margin)

        # --- DIM 0 (Z) and DIM 2 (X) : Force Square ---
        crop_z1, crop_z2, crop_x1, crop_x2 = _calculate_square_crop(
            dim0_min, dim0_max, dim0_limit,  # Z-axis parameters
            dim2_min, dim2_max, dim2_limit,  # X-axis parameters
            margin
        )

        return (int(crop_z1), int(crop_z2),
                int(crop_y1), int(crop_y2),
                int(crop_x1), int(crop_x2))
