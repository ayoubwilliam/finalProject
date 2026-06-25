"""
3D CT volume rotation using kornia.
Uses half-precision (float16) to reduce GPU memory by ~50%.
"""

import numpy as np
import torch
import kornia

import config as cfg


def rotate_ct_scan(data, angle_x, angle_y, angle_z):
    """Rotates a 3D CT volume by (angle_x, angle_y, angle_z) degrees using kornia on GPU."""
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data)
    else:
        data_tensor = data

    data_tensor = data_tensor.to(cfg.DEVICE)

    # half-precision reduces memory from ~0.5GB to ~0.25GB per volume
    data_tensor = data_tensor.half()

    # reshape: (X, Y, Z) → (Z, X, Y) → (1, 1, Z, X, Y) for kornia
    ct_tensor = data_tensor.permute(2, 0, 1)
    kornia_input_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

    # angles must match half-precision dtype
    angles_x_tensor = torch.tensor([angle_x], dtype=torch.float16, device=cfg.DEVICE)
    angles_y_tensor = torch.tensor([angle_y], dtype=torch.float16, device=cfg.DEVICE)
    angles_z_tensor = torch.tensor([angle_z], dtype=torch.float16, device=cfg.DEVICE)

    with torch.no_grad():
        rotated_tensor = kornia.geometry.transform.rotate3d(
            kornia_input_tensor,
            angles_x_tensor,
            angles_y_tensor,
            angles_z_tensor,
        )

    # back to float32 for downstream compatibility
    rotated_squeezed = rotated_tensor.squeeze(0).squeeze(0).float()
    final_tensor = rotated_squeezed.permute(1, 2, 0)

    return final_tensor
