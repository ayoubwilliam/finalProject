import kornia
import numpy as np
import sys
import torch

from file_handler import load_nifti, save_nifti
from constants import DEVICE

def rotate_ct_scan(data, angle_x: float, angle_y: float, angle_z: float):
    # 1. Move to GPU
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data)
    else:
        data_tensor = data

    data_tensor = data_tensor.to(DEVICE)

    # --- OPTIMIZATION: Convert to Float16 (Half Precision) ---
    # This reduces memory from ~0.5GB to ~0.25GB for the volume
    # And reduces the grid from ~1.6GB to ~0.8GB
    data_tensor = data_tensor.half() #todo check with itamar

    # 2. Permute and Reshape
    ct_tensor = data_tensor.permute(2, 0, 1) # (D, H, W)
    kornia_input_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

    # 3. Prepare angles (Must also be Float16/Half)
    angles_x_tensor = torch.tensor([angle_x], dtype=torch.float16, device=DEVICE)
    angles_y_tensor = torch.tensor([angle_y], dtype=torch.float16, device=DEVICE)
    angles_z_tensor = torch.tensor([angle_z], dtype=torch.float16, device=DEVICE)

    # 4. Apply 3D rotation
    with torch.no_grad(): # Ensure no gradients are stored
        rotated_tensor = kornia.geometry.transform.rotate3d(
            kornia_input_tensor,
            angles_x_tensor,
            angles_y_tensor,
            angles_z_tensor
        )

    # 5. Convert back to Float32 for Numpy compatibility and saving
    rotated_squeezed = rotated_tensor.squeeze(0).squeeze(0).float()
    final_tensor = rotated_squeezed.permute(1, 2, 0)

    return final_tensor



if __name__ == "__main__":
    data, affine, header = load_nifti(sys.argv[1])
    data = rotate_ct_scan(data, sys.argv[3], sys.argv[4], sys.argv[5])
    save_nifti(sys.argv[2], data, affine, header)
