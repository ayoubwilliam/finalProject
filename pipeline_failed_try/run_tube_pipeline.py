import os
import time
import numpy as np

import config as cfg
from nifti_io import load_nifti
from tube_pipeline import pipeline
from randomized_tube_values import get_randomized_tube_params

NUMBER_OF_PAIRS_TO_GENERATE = 15
OUTPUT_DIR = "./output/pipeline_output/pair"
# Using the fallback paths that were previously in basic tube.py
input_ct_path = "../ct/ct_file1.nii.gz"
input_seg_path = "./segs/ct_file1_full_airway_v2.nii.gz"

def get_filename_from_path(path: str) -> str:
    return os.path.basename(path).split('.')[0]

def get_pair_dir(pair_index: int, input_path: str, suffix: str) -> str:
    input_filename = get_filename_from_path(input_path)
    path = os.path.join(OUTPUT_DIR, f"{input_filename}_Pair{pair_index}_{suffix}")
    os.makedirs(path, exist_ok=True)
    return path + os.sep

def get_random_rotation_angles():
    """Generates random rotation angles based on config ranges."""
    x = np.random.uniform(-cfg.ROT_ANGLE_X_RANGE_DEG, cfg.ROT_ANGLE_X_RANGE_DEG)
    y = np.random.uniform(-cfg.ROT_ANGLE_Y_RANGE_DEG, cfg.ROT_ANGLE_Y_RANGE_DEG)
    z = np.random.uniform(-cfg.ROT_ANGLE_Z_RANGE_DEG, cfg.ROT_ANGLE_Z_RANGE_DEG)
    return [x, y, z]

if __name__ == '__main__':

    # Bypass requirement for CT_ORIGINAL_DIR during local testing if needed
    if getattr(cfg, 'CT_ORIGINAL_DIR', None) is None:
        cfg.CT_ORIGINAL_DIR = "./ct"
    
    print(f"--- Starting Tube Pipeline Execution ---")
    print(f"Target CT: {input_ct_path}")
    print(f"Target Seg: {input_seg_path}")

    start_time = time.time()

    print("Loading CT and segmentation mask...")
    


    ct_data, _, _ = load_nifti(input_ct_path)
    lung_mask, _, _ = load_nifti(input_seg_path)

    for pair_index in range(1, NUMBER_OF_PAIRS_TO_GENERATE + 1):
        print(f"\n======================================")
        print(f" Generating Tube Pair {pair_index} of {NUMBER_OF_PAIRS_TO_GENERATE}")
        print(f"======================================")

        # Generate unique parameters for the tube and random rotation angles
        tube_params = get_randomized_tube_params(lung_mask.astype(np.uint8))
        
        prior_angle = get_random_rotation_angles()
        current_angle = get_random_rotation_angles()

        has_prior_tube = np.random.random() < getattr(cfg, 'ADD_MASS_PRIOR_PROBABILITY', 0.5)
        has_current_tube = np.random.random() < getattr(cfg, 'ADD_MASS_CURRENT_PROBABILITY', 0.8)

        print(f"Prior Angles: {prior_angle} | Has Tube: {has_prior_tube}")
        print(f"Current Angles: {current_angle} | Has Tube: {has_current_tube}")

        # WITH CROP
        print("\n--- Running Variant: WITH CROP ---")
        pair_dir_crop = get_pair_dir(pair_index, input_ct_path, "crop")
        pipeline(
            pair_dir=pair_dir_crop,
            ct_data=ct_data,
            lungs_mask=lung_mask,
            prior_tube_params=tube_params,
            current_tube_params=tube_params, # Normally same tube params used for prior/current
            prior_angles=prior_angle,
            current_angles=current_angle,
            has_prior_tube=has_prior_tube,
            has_current_tube=has_current_tube,
            use_crop=True
        )

        # WITHOUT CROP
        print("\n--- Running Variant: WITHOUT CROP ---")
        pair_dir_no_crop = get_pair_dir(pair_index, input_ct_path, "no_crop")
        pipeline(
            pair_dir=pair_dir_no_crop,
            ct_data=ct_data,
            lungs_mask=lung_mask,
            prior_tube_params=tube_params,
            current_tube_params=tube_params,
            prior_angles=prior_angle,
            current_angles=current_angle,
            has_prior_tube=has_prior_tube,
            has_current_tube=has_current_tube,
            use_crop=False
        )

    end_time = time.time()
    print(f"\nDone with all Pairs!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
