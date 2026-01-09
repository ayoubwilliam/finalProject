import torch
from device_constants import DEVICE
from rotation import rotate_ct_scan


def get_segmentation_bounds(segmentation, angles, margin):
    """
    Rotates the segmentation on GPU, calculates the bounding box for X and Y,
    but keeps the FULL Depth (Z) axis.
    """
    with torch.no_grad():
        # 1. Load Segmentation to GPU
        if not isinstance(segmentation, torch.Tensor):
            seg_tensor = torch.from_numpy(segmentation).to(DEVICE)
        else:
            seg_tensor = segmentation.to(DEVICE)

        # Convert to float (half) for rotation
        seg_float = seg_tensor.half()

        # 2. Rotate
        rotated_seg = rotate_ct_scan(seg_float, angles[0], angles[1], angles[2])

        # Threshold back to binary
        rotated_seg = (rotated_seg > 0.5)

        # 3. Calculate Bounds
        non_zero_indices = torch.nonzero(rotated_seg)

        if non_zero_indices.numel() == 0:
            return None

        # Get min/max for Y and X only (ignore Z from indices)
        # Note: non_zero_indices is (N, 3) -> [z, y, x]
        _, y_min, x_min = non_zero_indices.min(dim=0).values
        _, y_max, x_max = non_zero_indices.max(dim=0).values

        # 4. Apply Margin & Clamp
        depth, height, width = rotated_seg.shape

        # --- Z AXIS: Keep FULL Depth ---
        z1 = 0
        z2 = depth

        # --- Y & X AXES: Crop with Margin ---
        y1 = max(0, y_min.item() - margin)
        x1 = max(0, x_min.item() - margin)

        y2 = min(height, y_max.item() + margin)
        x2 = min(width, x_max.item() + margin)

        # 5. Implicit Cleanup
        return (z1, z2, y1, y2, x1, x2)
