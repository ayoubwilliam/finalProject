import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from file_handler import load_nifti, save_nifti
from create_shapes import create_sphere, apply_mask
from bspline import bspline
from rotation import rotate_ct_scan
from pooling import apply_pooling, KERNEL_SIZE as POOL_KERNEL
from drr_with_post_processing import create_drr_from_ct

# ----- hyper-parameters -----
MASS_RADIUS_RANGE = (10, 20)  # in voxels
MASS_INTENSITY_RANGE = (-100, 100)  # HU range for consolidation-like mass
ROT_ANGLE_RANGE_DEG = 15.0  # sample angles in [-15, 15]

# Bspline
B_SPLINE_DEFORMATION = 0.04
B_SPLINE_GRID_DENSITY = 8

# Heatmap
MIN_INTENSITY_DIFF = 0.2  # Only meaningful changes


def sample_point_in_lungs(lung_mask: np.ndarray) -> tuple[int, int, int]:
    """
    This function samples one random voxel location inside a 3D lung mask.
        parameters:
        1) lung_mask: boolean or integer 3D array where lung voxels are nonzero.
        returns: point: tuple (x, y, z) of a randomly selected lung voxel.
    """
    # Collect coordinates of all voxels that belong to lungs
    coords = np.argwhere(lung_mask > 0)

    # Sample a random index from the available lung voxels
    idx = np.random.randint(0, len(coords))

    # Return the voxel coordinate as a tuple (x, y, z)
    return tuple(coords[idx])


def create_mass_mask_in_lungs(ct_data: np.ndarray, lung_mask: np.ndarray,
                              radius_range: tuple[int, int]) -> np.ndarray:
    """
    This function samples a lung voxel, creates a spherical mass around it
    and clips the sphere to the lung region.
        parameters:
        1) ct_data: 3D volume used only for spatial dimensions.
        2) lung_mask: 3D mask defining where lung voxels are valid.
        3) radius_range: (min_radius, max_radius) specifying sphere size range.
        returns: mass_mask: boolean 3D mask of a sphere restricted to lung voxels.
    """
    # Sample one valid location inside lungs for the sphere center
    cx, cy, cz = sample_point_in_lungs(lung_mask)

    # Sample radius from given range
    r_min, r_max = radius_range
    radius = np.random.randint(r_min, r_max + 1)

    # Create spherical mask around center
    sphere_mask = create_sphere(ct_data, [cx, cy, cz], radius)

    # Restrict sphere to lung region only
    mass_mask = sphere_mask & (lung_mask > 0)

    return mass_mask


def create_deformed_mass_volume(ct_shape: tuple[int, int, int], mass_mask: np.ndarray,
                                base_intensity: int, is_noised: bool,
                                grid_density_factor: int = B_SPLINE_GRID_DENSITY,
                                deformation_factor: float = B_SPLINE_DEFORMATION) -> np.ndarray:
    """
    This function generates a synthetic mass volume, fills it with intensity and noise inside the mask
    and applies a 3D B-spline deformation to create a realistic entity.
        parameters:
        1) ct_shape: shape of the target CT volume.
        2) mass_mask: boolean mask defining the initial spherical mass region.
        3) base_intensity: HU value assigned to the mass before deformation.
        4) is_noised: flag controlling whether random noise is added to the mass.
        5) grid_density_factor: density of the B-spline control grid.
        6) deformation_factor: amplitude of the deformation field.
        returns: deformed_mass: 3D float volume of the deformed synthetic entity.
    """
    # Build a zero-filled volume and apply the mask so the bspline can operate only on the region
    mass_volume = np.zeros(ct_shape, dtype=np.float32)
    mass_volume = apply_mask(mass_volume, mass_mask, base_intensity, is_noised)

    # Apply B-spline deformation to create a realistic mass shape
    deformed_mass = bspline(mass_volume, grid_density_factor, deformation_factor)

    # Convert to float32 for consistent CT volume typing
    return deformed_mass.astype(np.float32)


def insert_synthetic_structure(ct_data: np.ndarray,
                               deformed_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function inserts a synthetic structure into the CT volume by overwriting non-zero voxels.
       parameters:
       1) ct_data: original 3D CT volume.
       2) deformed_mass: 3D volume containing the deformed synthetic structure.
       returns:
       1) ct_with_mass: CT volume after embedding the synthetic structure.
       2) mass_mask: boolean mask indicating where the structure was applied.
    """
    # create a writable CT volume and identify where the synthetic structure exists
    ct_with_mass = ct_data.astype(np.float32)
    mass_mask = deformed_mass != 0

    # embed the synthetic structure by overwriting CT values inside its region
    ct_with_mass[mass_mask] = deformed_mass[mass_mask]

    return ct_with_mass, mass_mask


def smooth_mass_region(ct_with_mass: np.ndarray, mass_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """This function smooths only the mass region by applying pooling restricted to the provided mask"""
    return apply_pooling(ct_with_mass, mass_mask, kernel_size)


def get_random_rotation_angles(range_deg: float = ROT_ANGLE_RANGE_DEG) -> tuple[float, float, float]:
    """
    This function samples three independent rotation angles uniformly from the range [-range_deg, range_deg].
        parameters:
        1) range_deg: maximum absolute angle in degrees for each axis.
        returns:
        1) angle_x: rotation around x-axis
        2) angle_y: rotation around y-axis
        3) angle_z: rotation around z-axis
    """
    angles = np.random.uniform(-range_deg, range_deg, size=3)
    return float(angles[0]), float(angles[1]), float(angles[2])


def get_random_intensity() -> int:
    """ This function samples a random intensity value within the predefined mass intensity range """
    return int(np.random.uniform(MASS_INTENSITY_RANGE[0], MASS_INTENSITY_RANGE[1]))


def clip_synthetic_to_lungs(ct_with_mass: np.ndarray,
                            deformed_mask: np.ndarray,
                            lung_mask: np.ndarray,
                            original_ct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function removes synthetic regions outside the lungs and restores original CT values there.
    parameters:
    1) ct_with_mass: CT volume after inserting the synthetic structure.
    2) deformed_mask: boolean mask where the synthetic structure is present.
    3) lung_mask: boolean or integer mask defining the lung region.
    4) original_ct: original CT volume before inserting the synthetic structure.
    returns:
    1) clipped_ct: CT volume where the synthetic structure is confined to lungs.
    2) clipped_mask: boolean mask of the synthetic structure restricted to lungs.
    """
    # derive lung-only boolean mask
    lungs = lung_mask > 0

    # find synthetic voxels that leaked outside the lungs
    outside_lungs = np.logical_and(deformed_mask, np.logical_not(lungs))

    # restore original CT values outside lungs
    ct_with_mass[outside_lungs] = original_ct[outside_lungs]

    # keep synthetic mask only inside lungs
    clipped_mask = np.logical_and(deformed_mask, lungs)

    return ct_with_mass, clipped_mask


def run_pipeline(ct_path: str, lung_mask_path: str,
                 output_ct_path: str, output_drr_path: str, is_prev) -> tuple[int, int, int]:
    # Load data
    ct_data, ct_affine, ct_header = load_nifti(ct_path)
    lung_seg, _, _ = load_nifti(lung_mask_path)

    # Create mass
    mass_mask = create_mass_mask_in_lungs(ct_data, lung_seg > 0, MASS_RADIUS_RANGE)

    # Apply bspline
    mass_intensity = get_random_intensity()
    deformed_mass = create_deformed_mass_volume(ct_data.shape, mass_mask, mass_intensity, is_noised=True)

    # Embed the deformed synthetic structure into the CT volume
    ct_with_mass, deformed_mask = insert_synthetic_structure(ct_data, deformed_mass)
    ct_with_mass, deformed_mask = clip_synthetic_to_lungs(ct_with_mass, deformed_mask, lung_seg, ct_data)
    # todo: maybe fix ct/mask with bspline- clip and lung mask

    # Apply pooling
    deformed_mask = deformed_mask & (lung_seg > 0)  # restrict smoothing only to areas inside lungs
    ct_smoothed = smooth_mass_region(ct_with_mass, deformed_mask, POOL_KERNEL)

    # Rotate ct
    # angle_x, angle_y, angle_z = get_random_rotation_angles()
    if is_prev:
        angle_x, angle_y, angle_z = 30, 0, 0
    else:
        angle_x, angle_y, angle_z = -30, 0, 0

    ct_rotated = rotate_ct_scan(ct_smoothed, angle_x, angle_y, angle_z)

    # Save ct and create drr
    save_nifti(output_ct_path, ct_rotated, ct_affine, ct_header)
    # create_drr_from_ct(ct_rotated, output_drr_path, projection_axis=1)

    return angle_x, angle_y, angle_z


def create_heatmap(prior_drr_path: str,
                   prior_rotation_angles: tuple[float, float, float],
                   current_drr_path: str,
                   current_rotation_angles: tuple[float, float, float],
                   heatmap_path: str) -> None:
    """
        This function aligns a prior DRR to the current DRR's orientation and generates a difference heatmap
         where positive differences (growth) appear green and negative differences (shrinkage) appear red.
        parameters:
        1) prior_drr_path: path to the baseline DRR image
        2) prior_rotation_angles: rotation angles (x, y, z) of the baseline image
        3) current_drr_path: path to the current DRR image
        4) current_rotation_angles: rotation angles (x, y, z) of the current image
        5) heatmap_path: destination path for the generated heatmap
    """
    # Load and standardize images to 2D grayscale
    prior_img = plt.imread(prior_drr_path)
    current_img = plt.imread(current_drr_path)

    # Robustly flatten to 2D (H, W)
    if prior_img.ndim == 3:
        prior_img = prior_img[..., 0]
    if current_img.ndim == 3:
        current_img = current_img[..., 0]

    # Ensure no singleton dimensions remain (e.g., (H, W, 1) -> (H, W))
    prior_img = np.squeeze(prior_img)
    current_img = np.squeeze(current_img)

    # Compute rotation difference
    prior_theta, _, _ = prior_rotation_angles
    current_theta, _, _ = current_rotation_angles
    delta_angle = current_theta - prior_theta
    print(f"Delta Angle: {delta_angle}")

    # Align prior image to current orientation
    prior_aligned = rotate(prior_img, delta_angle, reshape=False, order=1, mode="nearest")
    # Match foreground regions of current DRR in the rotated prior
    prior_aligned[current_img == 1] = 1
    plt.imsave("pipeline/rotated_prior.png", prior_aligned, cmap="gray")

    # Compute difference map
    diff = current_img - prior_aligned

    # Stack 2D image to create 3D RGB heatmap base
    heatmap = np.dstack((current_img, current_img, current_img))

    # Apply Green overlay for positive difference (growth)
    pos_mask = diff > MIN_INTENSITY_DIFF
    heatmap[pos_mask, 1] += diff[pos_mask]
    heatmap[pos_mask, 0] -= diff[pos_mask] * 0.3
    heatmap[pos_mask, 2] -= diff[pos_mask] * 0.3

    # Apply Red overlay for negative difference (shrinkage)
    neg_mask = diff < -MIN_INTENSITY_DIFF
    abs_diff = np.abs(diff[neg_mask])
    heatmap[neg_mask, 0] += abs_diff
    heatmap[neg_mask, 1] -= abs_diff * 0.3
    heatmap[neg_mask, 2] -= abs_diff * 0.3

    # Save final heatmap
    heatmap = np.clip(heatmap, 0.0, 1.0)
    plt.imsave(heatmap_path, heatmap)


def create_synthetic_pair_and_heatmap(ct_path, lung_mask_path,
                                      prior_output_ct_path, current_output_ct_path,
                                      prior_output_drr_path, current_output_drr_path, heatmap_path):
    # Get current and prior ct scans
    prior_rotation_angles = run_pipeline(ct_path, lung_mask_path,
                                         prior_output_ct_path, prior_output_drr_path, True)
    # print(prior_rotation_angles)
    current_rotation_angles = run_pipeline(ct_path, lung_mask_path,
                                           current_output_ct_path, current_output_drr_path, False)
    # print(current_rotation_angles)

    # todo: rotate prior by current
    # todo: create drr of rotated prior and current
    # todo: create heatmap
    # todo: post processing of og prior and current
    # todo: show changes in heatmap with post processed current image

    # Create heatmap
    create_heatmap(prior_output_drr_path, prior_rotation_angles,
                   current_output_drr_path, current_rotation_angles,
                   heatmap_path)


if __name__ == "__main__":
    # Get arguments
    if len(sys.argv) != 8:
        raise RuntimeError("Invalid number of arguments.")
    ct_path = sys.argv[1]
    lung_mask_path = sys.argv[2]
    prior_output_ct_path = sys.argv[3]
    current_output_ct_path = sys.argv[4]
    prior_output_drr_path = sys.argv[5]
    current_output_drr_path = sys.argv[6]
    heatmap_path = sys.argv[7]

    create_synthetic_pair_and_heatmap(ct_path, lung_mask_path,
                                      prior_output_ct_path, current_output_ct_path,
                                      prior_output_drr_path, current_output_drr_path, heatmap_path)
