import kornia
import numpy as np
import sys
import torch

from file_handler import load_nifti, save_nifti


def rotate_ct_scan(data: np.array, angle_x: float, angle_y: float, angle_z: float) -> np.array:
    # Permute from (H, W, D) to (D, H, W)
    ct_tensor = torch.from_numpy(data).permute(2, 0, 1)

    # Reshape the tensor to Kornia's 5D format (B, C, D, H, W)- adding Batch and Channel dimension
    kornia_input_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

    # Prepare angle float tensors for rotate3d
    angles_x_tensor, angles_y_tensor, angles_z_tensor = [torch.tensor([float(a)], dtype=torch.float32)
                                                         for a in (angle_x, angle_y, angle_z)]

    # Apply the 3D rotation
    rotated_tensor = kornia.geometry.transform.rotate3d(kornia_input_tensor,
                                                        angles_x_tensor, angles_y_tensor, angles_z_tensor)

    # Convert the rotated tensor back to a Numpy array: removes the Batch and Channel dimensions (.cpu())
    rotated_numpy = rotated_tensor.squeeze().numpy()

    # Permute back from (D, H, W) to (H, W, D) for saving
    return np.transpose(rotated_numpy, (1, 2, 0))


if __name__ == "__main__":
    data, affine, header = load_nifti(sys.argv[1])
    data = rotate_ct_scan(data, sys.argv[3], sys.argv[4], sys.argv[5])
    save_nifti(sys.argv[2], data, affine, header)
