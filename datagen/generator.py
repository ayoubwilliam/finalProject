"""
Top-level data generation orchestrator.
Iterates all CT scans, samples random parameters, and calls the pipeline for each pair.
"""

import os
import random
import time
import numpy as np
import argparse

import config as cfg
from lib.nifti_io import load_nifti, create_lungs_seg_path
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


def create_pair(pair_dir, ct_data, lung_mask, trachea_mask):
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

    has_prior_tube = np.random.random() < cfg.ADD_TUBE_PRIOR_PROBABILITY
    has_current_tube = np.random.random() < cfg.ADD_TUBE_CURRENT_PROBABILITY

    from datagen.tube_randomization import get_random_tube_diameter
    tube_diameter, tube_thickness = get_random_tube_diameter()

    pipeline(pair_dir, ct_data, lung_mask, trachea_mask, radius,
             prior_pos, current_pos,
             prior_angle, current_angle,
             has_prior_mass=has_prior_mass,
             has_current_mass=has_current_mass,
             has_prior_tube=has_prior_tube,
             has_current_tube=has_current_tube,
             tube_diameter=tube_diameter,
             tube_thickness=tube_thickness)


def create_pairs_for_scan(input_path, lungs_seg_path, trachea_seg_path, filename):
    """Creates all pairs for a single CT scan."""
    print("\nCreating pairs for ", input_path)
    ct_data, _, _ = load_nifti(input_path)
    lungs_data, _, _ = load_nifti(lungs_seg_path)

    # Load trachea segmentation if tube is enabled
    trachea_data = None
    if cfg.ADD_TUBE:
        if trachea_seg_path and os.path.exists(trachea_seg_path):
            trachea_data, _, _ = load_nifti(trachea_seg_path)
        else:
            print(f"  Warning: Tube enabled but trachea segmentation not found at {trachea_seg_path}")

    for index in range(1, cfg.NUMBER_OF_PAIRS_PER_SCAN + 1):
        pair_dir = _get_pair_dir(index, filename)
        print("\nPair number: ", index)
        create_pair(pair_dir, ct_data, lungs_data, trachea_data)


def create_trachea_seg_path(filename):
    """Builds the trachea segmentation output path for a given CT filename."""
    os.makedirs(cfg.TRACHEA_SEGMENTATION_DIR, exist_ok=True)
    base = filename.split(cfg.NIFTI_EXTENSION)[0]
    return os.path.join(cfg.TRACHEA_SEGMENTATION_DIR, base + cfg.TRACHEA_SEG_SUFFIX + cfg.NIFTI_EXTENSION)


def create_pairs_for_all_scans():
    """Iterates all CT scans in ct_original, creates pairs for each."""

    # 1. Load and shuffle the file list to distribute work across multiple processes
    all_files = os.listdir(cfg.CT_ORIGINAL_DIR)
    random.shuffle(all_files)

    for filename in all_files:
        input_path = os.path.join(cfg.CT_ORIGINAL_DIR, filename)
        output_path = _create_output_path(filename)
        lungs_seg_path = create_lungs_seg_path(filename)
        trachea_seg_path = create_trachea_seg_path(filename)

        # 2. Atomic lock mechanism to prevent race conditions
        try:
            # Attempt to create target directory. exist_ok=False is critical here!
            os.makedirs(output_path, exist_ok=False)
        except FileExistsError:
            # Another process has already created this directory. Skip to next scan.
            print(f"Skipping {filename}, another process is already working on it.")
            continue

        # 3. Verify segmentations exist before starting the heavy pipeline
        if not os.path.exists(lungs_seg_path):
            print(f"Can't find {lungs_seg_path} file to load. Make sure to run seg_generator.py first"
                  f" to create the lungs segmentation!")
            continue

        elif not os.path.exists(trachea_seg_path):
            print(f"Can't find {trachea_seg_path} file to load. Make sure to run seg_generator.py first"
                  f" to create the trachea segmentation!")
            continue

        # Lock acquired successfully and segmentations exist. Proceed to pipeline.
        create_pairs_for_scan(input_path, lungs_seg_path, trachea_seg_path, filename)


# def create_pairs_for_all_scans():
#     """Iterates all CT scans in ct_original, creates pairs for each."""
#     for filename in os.listdir(cfg.CT_ORIGINAL_DIR):
#         input_path = os.path.join(cfg.CT_ORIGINAL_DIR, filename)
#         output_path = _create_output_path(filename)
#         lungs_seg_path = create_lungs_seg_path(filename)
#         trachea_seg_path = create_trachea_seg_path(filename)
#
#         if os.path.exists(output_path):
#             print(f"Output dir exists, skipping scan: {output_path}")
#             continue
#         elif not os.path.exists(lungs_seg_path):
#             print(f"Can't find {lungs_seg_path} file to load. Make sure to run seg_generator.py first"
#                   f" to create the lungs segmentation!")
#         elif not os.path.exists(trachea_seg_path):
#             print(f"Can't find {trachea_seg_path} file to load. Make sure to run seg_generator.py first"
#                   f" to create the trachea segmentation!")
#         else:
#             create_pairs_for_scan(input_path, lungs_seg_path, trachea_seg_path, filename)


def run_generator():
    parser = argparse.ArgumentParser(description="Run data generation directly.")
    parser.add_argument("ct_dir", type=str, help="Path to the directory containing original CT scans.")
    args = parser.parse_args()

    cfg.set_ct_input_dir(args.ct_dir)
    print(7)

    start_time = time.time()
    create_pairs_for_all_scans()
    end_time = time.time()
    print("\nDone with all Pairs for all scans!!!")
    print("Time elapsed: ", end_time - start_time, " seconds")
