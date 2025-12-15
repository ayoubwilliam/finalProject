import numpy as np
import gryds

from create_shapes import create_sphere, apply_mask
from file_handler import load_nifti, save_nifti
# from drr_with_post_processing import create_drr_with_processing
from rotation import rotate_ct_scan


def bspline(data, grid_density_factor, deformation_factor):
    data = data.astype(np.float32)
    x_shape = data.shape[0] // grid_density_factor
    y_shape = data.shape[1] // grid_density_factor
    z_shape = data.shape[2] // grid_density_factor
    gridx = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    gridy = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor
    gridz = np.random.rand(x_shape, y_shape, z_shape) * deformation_factor

    transform = gryds.BSplineTransformation([gridx, gridy, gridz])

    interpolator = gryds.Interpolator(data, order=3, mode="mirror")

    nifti_roi = interpolator.transform(transform)
    return nifti_roi


def add_deformed_sphere(data, intensity, pos, radius, grid_density_factor, deformation_factor):
    sphere = np.zeros_like(data, dtype=np.float32)
    sphere_mask = create_sphere(sphere, pos, radius)
    apply_mask(sphere, sphere_mask, intensity, True)

    deformed_sphere = bspline(sphere, grid_density_factor, deformation_factor)

    mask = np.round(deformed_sphere) != 0
    data[mask] = deformed_sphere[mask]

    return deformed_sphere, mask


def add_deformed_sphere_fast(data, intensity, pos, radius, margin, grid_density_factor, deformation_factor):
    size = 2 * radius + margin
    small_sphere_volume = np.zeros((size, size, size), dtype=np.float32)
    center = size // 2

    sphere_mask = create_sphere(small_sphere_volume, [center, center, center], radius)
    apply_mask(small_sphere_volume, sphere_mask, intensity, True)

    deformed_small_sphere = bspline(small_sphere_volume, grid_density_factor, deformation_factor)

    sphere = np.zeros_like(data, dtype=np.float32)

    x, y, z = pos
    half_size = size // 2
    sphere[x - half_size:x - half_size + size,
    y - half_size:y - half_size + size,
    z - half_size:z - half_size + size] = deformed_small_sphere

    mask = np.round(sphere) != 0
    data[mask] = sphere[mask]

    return sphere, mask


def main():
    input_path = "../../ct/1.3.6.1.4.1.14519.5.2.1.6279.6001.100332161840553388986847034053.nii.gz"
    intensity = 25
    pos = [180, 250, 300]
    radius = 20
    margin = 4 * radius
    grid_density_factor = 8
    deformation_factor = 0.01

    # Test normal method
    print("Running add_deformed_sphere...")
    data, affine, header = load_nifti(input_path)
    deformed_sphere, mask = add_deformed_sphere(data, intensity, pos, radius, grid_density_factor,
                                                deformation_factor)
    save_nifti("./shapes_output/bspline_normal.nii.gz", data, affine, header)
    save_nifti("./shapes_output/bspline_normal_mask.nii.gz", deformed_sphere, affine, header)

    deformation_factor = 0.08
    # Test fast method
    print("Running add_deformed_sphere_fast...")
    data, affine, header = load_nifti(input_path)
    deformed_sphere, mask = add_deformed_sphere_fast(data, intensity, pos, radius, margin,
                                                     grid_density_factor, deformation_factor)
    save_nifti("./shapes_output/bspline_fast.nii.gz", data, affine, header)
    save_nifti("./shapes_output/bspline_fast_mask.nii.gz", deformed_sphere, affine, header)

    print("Done!")


if __name__ == '__main__':
    main()
