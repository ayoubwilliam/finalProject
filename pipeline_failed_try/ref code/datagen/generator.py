"""
Top-level data generation orchestrator.
Iterates all CT scans, samples random parameters, and calls the pipeline for each pair.
"""

import os
import time
import numpy as np

import config as cfg
from lib.nifti_io import load_nifti, create_seg_path
from datagen.pipeline import pipeline


def get_random_radius():
    """Samples a random integer radius in [RADIUS_MIN, RADIUS_MAX]."""
    return np.random.randint(cfg.RADIUS_MIN, cfg.RADIUS_MAX + 1)


def sample_point_in_lungs(coords):
    """Picks a random voxel coordinate from the lung mask."""
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


def create_pair(pair_dir, ct_data, lung_mask):
    """Generates a single prior-current pair with random mass positions and rotations."""
    radius = get_random_radius()

    coords = np.argwhere(lung_mask > 0)
    if len(coords) == 0:
        return
    prior_pos = sample_point_in_lungs(coords)
    current_pos = sample_point_in_lungs(coords)

    prior_angle = get_random_rotation_angles()
    current_angle = get_random_rotation_angles()

    has_prior_mass = np.random.random() < cfg.ADD_MASS_PRIOR_PROBABILITY
    has_current_mass = np.random.random() < cfg.ADD_MASS_CURRENT_PROBABILITY

    pipeline(pair_dir, ct_data, lung_mask, radius,
             prior_pos, current_pos,
             prior_angle, current_angle,
             has_prior_mass=has_prior_mass,
             has_current_mass=has_current_mass)


def create_pairs_for_scan(input_path, seg_path, filename):
    """Creates all pairs for a single CT scan."""
    print("\nCreating pairs for ", input_path)
    ct_data, _, _ = load_nifti(input_path)
    seg_data, _, _ = load_nifti(seg_path)

    for index in range(1, cfg.NUMBER_OF_PAIRS_PER_SCAN + 1):
        pair_dir = _get_pair_dir(index, filename)
        print("\nPair number: ", index)
        create_pair(pair_dir, ct_data, seg_data)


def create_pairs_for_all_scans():
    """Iterates all CT scans in ct_original, creates pairs for each."""
    for filename in os.listdir(cfg.CT_ORIGINAL_DIR):
        input_path = os.path.join(cfg.CT_ORIGINAL_DIR, filename)
        output_path = _create_output_path(filename)
        seg_path = create_seg_path(filename)

        if os.path.exists(output_path):
            print(f"Output dir exists, skipping scan: {output_path}")
            continue
        elif not os.path.exists(seg_path):
            print(f"Can't find {seg_path} file to load. Make sure to run seg_generator.py first"
                  f" to create the lungs segmentation!")
        else:
            create_pairs_for_scan(input_path, seg_path, filename)


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
