"""
Top-level data generation orchestrator.
Iterates all CT scans, samples random ET-tube parameters, and calls the pipeline for each pair.

Tube placement is constrained to the TRACHEA mask (sampled from trachea voxels),
while the full lungs+trachea mask is used for cropping and boundary correction.
"""

import os
import time
import numpy as np

import config as cfg
from lib.nifti_io import load_nifti, create_seg_path, create_trachea_seg_path
from datagen.pipeline import pipeline


def get_random_tube_params():
    """
    Samples random ET-tube geometry parameters.

    Returns:
        tube_length:  int — length of tube in voxels.
        outer_radius: int — outer wall radius in voxels.
        inner_radius: int — inner lumen radius in voxels (always < outer).
    """
    tube_length = np.random.randint(cfg.TUBE_LENGTH_MIN, cfg.TUBE_LENGTH_MAX + 1)
    outer_radius = np.random.randint(cfg.TUBE_OUTER_RADIUS_MIN, cfg.TUBE_OUTER_RADIUS_MAX + 1)
    inner_radius = np.random.randint(cfg.TUBE_INNER_RADIUS_MIN,
                                     min(cfg.TUBE_INNER_RADIUS_MAX, outer_radius - 1) + 1)
    return tube_length, outer_radius, inner_radius


def sample_point_in_segmentation(coords):
    """Picks a random voxel coordinate from the given coordinate array."""
    idx = np.random.randint(0, len(coords))
    return tuple(coords[idx])


def get_random_rotation_angles():
    """Samples three independent rotation angles uniformly from [-range, +range]."""
    angle_x = np.random.uniform(-cfg.ROT_ANGLE_X_RANGE_DEG, cfg.ROT_ANGLE_X_RANGE_DEG)
    angle_y = np.random.uniform(-cfg.ROT_ANGLE_Y_RANGE_DEG, cfg.ROT_ANGLE_Y_RANGE_DEG)
    angle_z = np.random.uniform(-cfg.ROT_ANGLE_Z_RANGE_DEG, cfg.ROT_ANGLE_Z_RANGE_DEG)
    return float(angle_x), float(angle_y), float(angle_z)


def _remove_extension(filename):
    """Strips the .nii.gz extension from a filename."""
    return filename.split(cfg.NIFTI_EXTENSION)[0]


def _create_output_path(filename):
    """Builds the output directory path for a given CT filename."""
    base = _remove_extension(filename)
    return os.path.join(cfg.GENERATED_SYNTHETIC_DIR, base)


def _get_pair_dir(pair_index, filename):
    """Creates and returns the directory path for a specific pair."""
    output_path = _create_output_path(filename)
    path = os.path.join(output_path, f"Pair{pair_index}")
    os.makedirs(path, exist_ok=True)
    return path + os.sep


def create_pair(pair_dir, ct_data, seg_mask, trachea_mask):
    """
    Generates a single prior-current pair with random tube positions and rotations.

    Tube position is sampled from the TRACHEA mask (anatomically correct placement).
    The full seg_mask (lungs + trachea) is passed to the pipeline for cropping
    and boundary correction.
    """
    tube_length, outer_radius, inner_radius = get_random_tube_params()

    # Sample tube placement from trachea voxels only
    trachea_coords = np.argwhere(trachea_mask > 0)
    if len(trachea_coords) == 0:
        print("WARNING: Trachea mask is empty. Falling back to full segmentation mask.")
        trachea_coords = np.argwhere(seg_mask > 0)
    if len(trachea_coords) == 0:
        return

    prior_pos = sample_point_in_segmentation(trachea_coords)
    current_pos = sample_point_in_segmentation(trachea_coords)

    prior_angle = get_random_rotation_angles()
    current_angle = get_random_rotation_angles()

    has_prior_mass = np.random.random() < cfg.ADD_MASS_PRIOR_PROBABILITY
    has_current_mass = np.random.random() < cfg.ADD_MASS_CURRENT_PROBABILITY

    # Pass full seg_mask (lungs + trachea) to pipeline for cropping and mask correction
    pipeline(pair_dir, ct_data, seg_mask,
             tube_length, outer_radius, inner_radius,
             prior_pos, current_pos,
             prior_angle, current_angle,
             has_prior_mass=has_prior_mass,
             has_current_mass=has_current_mass)


def create_pairs_for_scan(input_path, seg_path, trachea_path, filename):
    """Creates all pairs for a single CT scan."""
    print("\nCreating pairs for ", input_path)
    ct_data, _, _ = load_nifti(input_path)
    seg_data, _, _ = load_nifti(seg_path)
    trachea_data, _, _ = load_nifti(trachea_path)

    for index in range(1, cfg.NUMBER_OF_PAIRS_PER_SCAN + 1):
        pair_dir = _get_pair_dir(index, filename)
        print("\nPair number: ", index)
        create_pair(pair_dir, ct_data, seg_data, trachea_data)


def create_pairs_for_all_scans():
    """Iterates all CT scans in ct_original, creates pairs for each."""
    for filename in os.listdir(cfg.CT_ORIGINAL_DIR):
        input_path = os.path.join(cfg.CT_ORIGINAL_DIR, filename)
        output_path = _create_output_path(filename)
        seg_path = create_seg_path(filename)
        trachea_path = create_trachea_seg_path(filename)

        if os.path.exists(output_path):
            print(f"Output dir exists, skipping scan: {output_path}")
            continue
        elif not os.path.exists(seg_path):
            print(f"Can't find {seg_path} file to load. Make sure to run seg_generator.py first"
                  f" to create the lungs segmentation!")
        elif not os.path.exists(trachea_path):
            print(f"Can't find {trachea_path} file to load. Make sure to run seg_generator.py first"
                  f" to create the trachea segmentation!")
        else:
            create_pairs_for_scan(input_path, seg_path, trachea_path, filename)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data generation directly.")
    parser.add_argument("ct_dir", type=str, help="Path to the directory containing original CT scans.")
    args = parser.parse_args()
    
    cfg.set_ct_input_dir(args.ct_dir)

    start_time = time.time()
    create_pairs_for_all_scans()
    end_time = time.time()
    print("\nDone with all Pairs for all scans!!!")
    print("Time elapsed: ", end_time - start_time, " seconds")
