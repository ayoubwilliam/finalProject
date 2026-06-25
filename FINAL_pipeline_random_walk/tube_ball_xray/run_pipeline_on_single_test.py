"""
Single-CT test runner for the combined tube + ball pipeline.
Based on the basic tube.py workflow:
  1. Check for cached lung + trachea segmentations → generate missing ones
  2. Run the pair-generation pipeline

Usage: just click Run (or: python run_pipeline_on_single_test.py)
"""

import os
import time
import numpy as np
import nibabel as nib

import config as cfg
from lib.nifti_io import load_nifti
from segmentation.tube_seg import generate_full_airway_mask
from datagen.pipeline import pipeline
from datagen.generator import get_random_radius, sample_point_in_lungs, get_random_rotation_angles

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
NUMBER_OF_PAIRS_TO_GENERATE = 15

INPUT_CT_PATH = "../ct/train_10712_a_5.nii.gz"
SEGS_DIR = "./segs"
OUTPUT_DIR = "../pipeline_output/"

# Lung segmentation path (same naming as xray_change_detection)
LUNG_SEG_FILENAME = os.path.basename(INPUT_CT_PATH).replace(".nii.gz", "").replace(".nii", "") + "_lungs_seg.nii.gz"
LUNG_SEG_PATH = os.path.join(SEGS_DIR, LUNG_SEG_FILENAME)

# Trachea segmentation path (same naming as tube_test6)
TRACHEA_SEG_FILENAME = os.path.basename(INPUT_CT_PATH).replace(".nii.gz", "").replace(".nii", "") + "_full_airway_v2.nii.gz"
TRACHEA_SEG_PATH = os.path.join(SEGS_DIR, TRACHEA_SEG_FILENAME)


# ==========================================
# SEGMENTATION HELPERS
# ==========================================
def ensure_lung_segmentation():
    """Check for cached lung segmentation, generate if missing."""
    if os.path.exists(LUNG_SEG_PATH):
        print(f"Found cached lung segmentation at {LUNG_SEG_PATH}")
        return LUNG_SEG_PATH

    print(f"Lung segmentation not found. Generating...")
    import tempfile
    from totalsegmentator.python_api import totalsegmentator
    from lib.nifti_io import merge_nifti

    os.makedirs(SEGS_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        totalsegmentator(
            input=INPUT_CT_PATH,
            output=tmp_dir,
            task=cfg.SEG_TASK,
            fast=True,
            preview=False,
            roi_subset=cfg.ROI_SUBSET,
            nr_thr_saving=1,
        )
        lobes = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
        merge_nifti(LUNG_SEG_PATH, *lobes)

    print(f"Lung segmentation saved to {LUNG_SEG_PATH}")
    return LUNG_SEG_PATH


def ensure_trachea_segmentation():
    """Check for cached trachea segmentation, generate if missing."""
    return generate_full_airway_mask(INPUT_CT_PATH, SEGS_DIR)


# ==========================================
# HELPERS
# ==========================================
def get_filename_from_path(path: str) -> str:
    return os.path.basename(path).split('.')[0]


def get_pair_dir(pair_index: int, input_path: str, suffix: str) -> str:
    input_filename = get_filename_from_path(input_path)
    path = os.path.join(OUTPUT_DIR, f"{input_filename}_Pair{pair_index}_{suffix}")
    os.makedirs(path, exist_ok=True)
    return path + os.sep


# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':

    # Bypass requirement for CT_ORIGINAL_DIR during local testing
    if cfg.CT_ORIGINAL_DIR is None:
        cfg.CT_ORIGINAL_DIR = "./ct"

    print(f"--- Starting Pipeline Execution ---")
    print(f"Target CT: {INPUT_CT_PATH}")
    print(f"ADD_TUBE: {cfg.ADD_TUBE}")
    print(f"ADD_DEFORMED_MASS: {cfg.ADD_DEFORMED_MASS}")

    start_time = time.time()

    # Check if CT exists
    if not os.path.exists(INPUT_CT_PATH):
        print(f"ERROR: CT file not found at {INPUT_CT_PATH}")
        exit(1)

    # --- Ensure segmentations exist ---
    print("\n--- Checking Segmentations ---")
    lung_seg_path = ensure_lung_segmentation()
    
    trachea_seg_path = None
    if cfg.ADD_TUBE:
        trachea_seg_path = ensure_trachea_segmentation()

    # --- Load data ---
    print("\nLoading CT and segmentation data...")
    ct_data, _, _ = load_nifti(INPUT_CT_PATH)
    lung_mask, _, _ = load_nifti(lung_seg_path)

    trachea_mask = None
    if cfg.ADD_TUBE and trachea_seg_path:
        trachea_mask = nib.load(trachea_seg_path).get_fdata().astype(np.uint8)

    # --- Generate pairs ---
    for pair_index in range(1, NUMBER_OF_PAIRS_TO_GENERATE + 1):
        print(f"\n======================================")
        print(f" Generating Pair {pair_index} of {NUMBER_OF_PAIRS_TO_GENERATE}")
        print(f"======================================")

        radius = get_random_radius()
        coords = np.argwhere(lung_mask > 0)

        if len(coords) == 0:
            print("ERROR: Lung mask is completely empty. Skipping...")
            continue

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

        print(f"Radius: {radius}")
        print(f"Prior Pos: {prior_pos} | Angles: {prior_angle} | Has Mass: {has_prior_mass} | Has Tube: {has_prior_tube}")
        print(f"Current Pos: {current_pos} | Angles: {current_angle} | Has Mass: {has_current_mass} | Has Tube: {has_current_tube}")
        print(f"Tube Diameter: {tube_diameter:.1f} | Tube Thickness: {tube_thickness}")

        # WITH CROP
        print("\n--- Running Variant: WITH CROP ---")
        pair_dir_crop = get_pair_dir(pair_index, INPUT_CT_PATH, "crop")
        pipeline(
            pair_dir=pair_dir_crop,
            ct_data=ct_data,
            lungs_mask=lung_mask,
            trachea_mask=trachea_mask,
            radius=radius,
            prior_pos=prior_pos,
            current_pos=current_pos,
            prior_angles=prior_angle,
            current_angles=current_angle,
            has_prior_mass=has_prior_mass,
            has_current_mass=has_current_mass,
            has_prior_tube=has_prior_tube,
            has_current_tube=has_current_tube,
            tube_diameter=tube_diameter,
            tube_thickness=tube_thickness,
            use_crop=True
        )

    end_time = time.time()
    print(f"\nDone with all Pairs!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
