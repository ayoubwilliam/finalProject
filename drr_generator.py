import numpy as np
from file_handler import load_nifti

from pipeline2 import pipeline

# data paths
INPUT_PATH = "ct/ct_file1.nii"
SEG_PATH = "ct/lungs.nii.gz"

# randomization parameters
R_MIN = 20
R_MAX = 40
ROT_ANGLE_RANGE_DEG = 15.0  # sample angles in [-15, 15]

NUMBER_OF_PAIRS = 10


def get_random_radius(r_min=R_MIN, r_max=R_MAX):
    return np.random.randint(r_min, r_max + 1)


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


def create_pair(pair_index):
    print("\nPair number: ", pair_index)
    # load lungs seg
    lung_mask, _, _ = load_nifti(SEG_PATH)

    # get random radius
    radius = get_random_radius()

    # sample valid locations inside lungs for the spheres' center
    prior_pos = sample_point_in_lungs(lung_mask)
    current_pos = sample_point_in_lungs(lung_mask)

    # get random rotation angles for prior and current
    prior_angle = get_random_rotation_angles()
    current_angle = get_random_rotation_angles()

    # run pipeline for prior and current
    pipeline(pair_index, INPUT_PATH, SEG_PATH, radius,
             prior_pos, current_pos,
             prior_angle, current_angle)


if __name__ == '__main__':
    for index in range(1, NUMBER_OF_PAIRS + 1):
        create_pair(index)
    print("\nDone with all Pairs!!!")
