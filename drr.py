import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from file_handler import load_nifti


def create_drr_from_ct(ct_data: np.array, output_path: str, projection_axis: int = 1) -> None:
    # todo: Clamp negative Hounsfield Unit (HU) values to remove air regions
    ct_data_clamped = np.clip(ct_data, -1000, None)

    # Sum along the chosen axis to simulate X-ray projection (DRR generation)
    drr_image = np.sum(ct_data_clamped, axis=projection_axis)

    # Normalize the image for display: This scales the image from 0 to 1
    if drr_image.max() > 0:
        drr_image = (drr_image - drr_image.min()) / (drr_image.max() - drr_image.min())

    # Rotate image 90° to correct flipped orientation from CT coordinate system
    drr_image = np.flip(drr_image.T, axis=0) #np.rot90(drr_image)


    # Save the resulting DRR image as grayscale PNG
    plt.imsave(output_path, drr_image, cmap='gray')


if __name__ == "__main__":
    # Define input and output paths
    data, affine, header = load_nifti(sys.argv[1])
    output_path = sys.argv[2]

    # Creating drr in coronal/sagittal/axial plane
    create_drr_from_ct(data, os.path.join(output_path, "drr_sagittal_plane.png"), 0)
    create_drr_from_ct(data, os.path.join(output_path, "drr_coronal_plane.png"), 1)
    create_drr_from_ct(data, os.path.join(output_path, "drr_axios_plane.png"), 2)
