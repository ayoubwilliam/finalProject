import os
import time
import numpy as np

import config as cfg
from lib.nifti_io import load_nifti
from datagen.pipeline import pipeline
from datagen.generator import get_random_radius, sample_point_in_lungs, get_random_rotation_angles

NUMBER_OF_PAIRS_TO_GENERATE = 3
OUTPUT_DIR = "../pipeline_output/"
input_ct_path = "../ct/train_10270_a_2.nii.gz"
input_seg_path = "../ct/train_10270_a_2_lungs_seg.nii.gz"

def get_filename_from_path(path: str) -> str:
    return os.path.basename(path).split('.')[0]

def get_pair_dir(pair_index: int, input_path: str, suffix: str) -> str:
    input_filename = get_filename_from_path(input_path)
    path = os.path.join(OUTPUT_DIR, f"{input_filename}_Pair{pair_index}_{suffix}")
    os.makedirs(path, exist_ok=True)
    return path + os.sep

if __name__ == '__main__':

    
    # Bypass requirement for CT_ORIGINAL_DIR during local testing
    if cfg.CT_ORIGINAL_DIR is None:
        cfg.CT_ORIGINAL_DIR = "./ct"
    
    print(f"--- Starting Pipeline Execution ---")
    print(f"Target CT: {input_ct_path}")
    print(f"Target Seg: {input_seg_path}")

    start_time = time.time()

    print("Loading segmentation mask for point sampling...")
    
    # Check if paths exist to prevent hard crash locally
    if not os.path.exists(input_ct_path) or not os.path.exists(input_seg_path):
        print("ERROR: Test CT files not found. Please create a 'ct' folder with test files or update the paths.")
        exit(1)

    ct_data, _, _ = load_nifti(input_ct_path)
    lung_mask, _, _ = load_nifti(input_seg_path)

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

        print(f"Radius: {radius}")
        print(f"Prior Pos: {prior_pos} | Angles: {prior_angle} | Has Mass: {has_prior_mass}")
        print(f"Current Pos: {current_pos} | Angles: {current_angle} | Has Mass: {has_current_mass}")

        # WITH CROP
        print("\n--- Running Variant: WITH CROP ---")
        pair_dir_crop = get_pair_dir(pair_index, input_ct_path, "crop")
        pipeline(
            pair_dir=pair_dir_crop,
            ct_data=ct_data,
            lungs_mask=lung_mask,
            radius=radius,
            prior_pos=prior_pos,
            current_pos=current_pos,
            prior_angles=prior_angle,
            current_angles=current_angle,
            has_prior_mass=has_prior_mass,
            has_current_mass=has_current_mass,
            use_crop=True
        )

        # WITHOUT CROP
        print("\n--- Running Variant: WITHOUT CROP ---")
        pair_dir_no_crop = get_pair_dir(pair_index, input_ct_path, "no_crop")
        pipeline(
            pair_dir=pair_dir_no_crop,
            ct_data=ct_data,
            lungs_mask=lung_mask,
            radius=radius,
            prior_pos=prior_pos,
            current_pos=current_pos,
            prior_angles=prior_angle,
            current_angles=current_angle,
            has_prior_mass=has_prior_mass,
            has_current_mass=has_current_mass,
            use_crop=False
        )

    end_time = time.time()
    print(f"\nDone with all Pairs!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
