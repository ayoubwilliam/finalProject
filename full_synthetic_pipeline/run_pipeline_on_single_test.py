"""
Single-scan test runner for the ET-tube synthetic data generation pipeline.

Runs the complete pipeline on a single CT scan:
  1. Segmentation (lung lobes + trachea) — if not already available
  2. ET-tube data generation (multiple prior/current pairs)

Tube placement is constrained to the TRACHEA mask for anatomical accuracy.
The full lungs+trachea mask is used for cropping and boundary correction.

Usage:
    python run_pipeline_on_single_test.py

Outputs are written to OUTPUT_DIR with both cropped and uncropped variants.
"""

import os
import sys
import gc
import time
import tempfile
import shutil
import numpy as np
import torch

import config as cfg
from lib.nifti_io import load_nifti, merge_nifti, save_nifti, create_seg_path
from datagen.pipeline import pipeline
from datagen.generator import (
    get_random_tube_params,
    sample_point_in_segmentation,
    get_random_rotation_angles,
)

# ==============================================================================
# CONFIGURATION — edit these paths for your local test setup
# ==============================================================================

NUMBER_OF_PAIRS_TO_GENERATE = 3
OUTPUT_DIR = "../pipeline_output_tube/"

# Path to the input CT scan (NIfTI format)
input_ct_path = "../ct_tube/train_10270_a_2.nii.gz"

# Paths to segmentation masks — if these files don't exist, segmentation
# will run automatically and save the results here.
input_seg_path = "../ct_tube/train_10270_a_2_lungs_seg.nii.gz"
input_trachea_path = "../ct_tube/train_10270_a_2_trachea_seg.nii.gz"

# Reproducibility seed (set to None for random each run)
RNG_SEED = 42


# ==============================================================================
# HELPERS
# ==============================================================================

def get_filename_from_path(path: str) -> str:
    return os.path.basename(path).split('.')[0]


def get_pair_dir(pair_index: int, input_path: str, suffix: str) -> str:
    input_filename = get_filename_from_path(input_path)
    path = os.path.join(OUTPUT_DIR, f"{input_filename}_Pair{pair_index}_{suffix}")
    os.makedirs(path, exist_ok=True)
    return path + os.sep


# ==============================================================================
# STAGE 1: SEGMENTATION
# ==============================================================================

def run_segmentation_for_scan(ct_path, seg_path, trachea_path):
    """
    Runs TotalSegmentator to produce:
      1. A merged lung+trachea segmentation mask (for cropping/boundary correction)
      2. A trachea-only mask (for tube placement)

    Skips if both files already exist.
    Returns (seg_path, trachea_path).
    """
    if os.path.exists(seg_path) and os.path.exists(trachea_path):
        print(f"[SEG] Segmentation already exists: {seg_path}")
        print(f"[SEG] Trachea mask already exists: {trachea_path}")
        return seg_path, trachea_path

    print(f"[SEG] Running TotalSegmentator on: {ct_path}")
    print(f"[SEG] ROI subset: {cfg.ROI_SUBSET}")

    # Import here to avoid loading TotalSegmentator when not needed
    from totalsegmentator.python_api import totalsegmentator

    # Free GPU memory before segmentation to reduce peak RAM usage
    gc.collect()
    torch.cuda.empty_cache()

    # Use custom temp directory to avoid filling up system /tmp
    custom_tmp_dir = os.path.join(cfg.DATA_DIR, "tmp_totalseg")
    os.makedirs(custom_tmp_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="totseg_single_", dir=custom_tmp_dir)
    try:
        try:
            totalsegmentator(
                input=ct_path,
                output=tmp_dir,
                task=cfg.SEG_TASK,
                fast=True,
                preview=False,
                roi_subset=cfg.ROI_SUBSET,
            )
        except Exception as e:
            import traceback
            if os.path.exists(tmp_dir) and len(os.listdir(tmp_dir)) > 0:
                print(f"[SEG] Warning: TotalSegmentator threw {type(e).__name__} during cleanup. Ignoring since outputs exist.")
            else:
                print(f"[SEG] ERROR: TotalSegmentator failed!")
                traceback.print_exc()
                sys.exit(1)

        # --- Save trachea mask separately BEFORE merging ---
        trachea_file = os.path.join(tmp_dir, "trachea.nii.gz")
        trachea_dir = os.path.dirname(trachea_path) if os.path.dirname(trachea_path) else "."
        os.makedirs(trachea_dir, exist_ok=True)

        if os.path.exists(trachea_file):
            shutil.copy2(trachea_file, trachea_path)
            print(f"[SEG] Trachea mask saved: {trachea_path}")
        else:
            print("[SEG] WARNING: trachea.nii.gz not found in TotalSegmentator output!")
            # Create empty trachea mask as fallback
            nii_files = [f for f in os.listdir(tmp_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
            if nii_files:
                data, affine, header = load_nifti(os.path.join(tmp_dir, nii_files[0]))
                empty = np.zeros_like(data)
                save_nifti(trachea_path, empty, affine, header)

        # --- Merge all masks (lungs + trachea) into single binary mask ---
        lobe_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        if not lobe_files:
            print("[SEG] ERROR: TotalSegmentator produced no output files.")
            sys.exit(1)

        seg_dir = os.path.dirname(seg_path) if os.path.dirname(seg_path) else "."
        os.makedirs(seg_dir, exist_ok=True)
        merge_nifti(seg_path, *lobe_files)
        print(f"[SEG] Merged segmentation saved: {seg_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Cleanup outer custom temp directory
    if os.path.exists(custom_tmp_dir):
        shutil.rmtree(custom_tmp_dir, ignore_errors=True)

    return seg_path, trachea_path


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':

    # Bypass requirement for CT_ORIGINAL_DIR during local testing
    if cfg.CT_ORIGINAL_DIR is None:
        cfg.CT_ORIGINAL_DIR = "./ct_tube"

    # Set RNG seed for reproducibility
    if RNG_SEED is not None:
        np.random.seed(RNG_SEED)
        print(f"[INIT] RNG seed set to {RNG_SEED}")

    print("=" * 60)
    print(" ET-TUBE SYNTHETIC DATA GENERATION — SINGLE SCAN TEST")
    print("=" * 60)
    print(f"  Target CT:      {input_ct_path}")
    print(f"  Target Seg:     {input_seg_path}")
    print(f"  Target Trachea: {input_trachea_path}")
    print(f"  Output Dir:     {OUTPUT_DIR}")
    print(f"  Pairs:          {NUMBER_OF_PAIRS_TO_GENERATE}")
    print(f"  Device:         {cfg.DEVICE}")
    print()

    start_time = time.time()

    # ------------------------------------------------------------------
    # STAGE 1: Segmentation
    # ------------------------------------------------------------------
    print("-" * 60)
    print(" STAGE 1: SEGMENTATION (lungs + trachea)")
    print("-" * 60)

    if not os.path.exists(input_ct_path):
        print(f"ERROR: CT file not found: {input_ct_path}")
        sys.exit(1)

    input_seg_path, input_trachea_path = run_segmentation_for_scan(
        input_ct_path, input_seg_path, input_trachea_path
    )

    # ------------------------------------------------------------------
    # STAGE 2: Load data
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print(" STAGE 2: LOADING CT + SEGMENTATION + TRACHEA")
    print("-" * 60)

    print("[LOAD] Loading CT volume...")
    ct_data, _, _ = load_nifti(input_ct_path)
    print(f"[LOAD] CT shape: {ct_data.shape}")

    print("[LOAD] Loading segmentation mask (lungs + trachea)...")
    seg_mask, _, _ = load_nifti(input_seg_path)
    print(f"[LOAD] Segmentation shape: {seg_mask.shape}")

    print("[LOAD] Loading trachea mask...")
    trachea_mask, _, _ = load_nifti(input_trachea_path)
    print(f"[LOAD] Trachea shape: {trachea_mask.shape}")

    seg_coords = np.argwhere(seg_mask > 0)
    trachea_coords = np.argwhere(trachea_mask > 0)
    print(f"[LOAD] Full seg non-zero voxels:  {len(seg_coords):,}")
    print(f"[LOAD] Trachea non-zero voxels:   {len(trachea_coords):,}")

    if len(seg_coords) == 0:
        print("ERROR: Segmentation mask is completely empty.")
        sys.exit(1)

    if len(trachea_coords) == 0:
        print("WARNING: Trachea mask is empty! Falling back to full segmentation for tube placement.")
        trachea_coords = seg_coords

    # ------------------------------------------------------------------
    # STAGE 3: Generate pairs
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print(" STAGE 3: ET-TUBE DATA GENERATION")
    print("-" * 60)

    for pair_index in range(1, NUMBER_OF_PAIRS_TO_GENERATE + 1):
        print(f"\n{'=' * 50}")
        print(f"  Generating Pair {pair_index} of {NUMBER_OF_PAIRS_TO_GENERATE}")
        print(f"{'=' * 50}")

        # --- Sample random tube parameters ---
        tube_length, outer_radius, inner_radius = get_random_tube_params()

        # Sample tube position from TRACHEA voxels (anatomically correct)
        prior_pos = sample_point_in_segmentation(trachea_coords)
        current_pos = sample_point_in_segmentation(trachea_coords)

        prior_angle = get_random_rotation_angles()
        current_angle = get_random_rotation_angles()

        has_prior_mass = np.random.random() < cfg.ADD_MASS_PRIOR_PROBABILITY
        has_current_mass = np.random.random() < cfg.ADD_MASS_CURRENT_PROBABILITY

        print(f"  [TUBE] Length: {tube_length}  |  Outer R: {outer_radius}  |  Inner R: {inner_radius}")
        print(f"  [POS]  Prior: {prior_pos}  |  Current: {current_pos}")
        print(f"  [ROT]  Prior: ({prior_angle[0]:.1f}, {prior_angle[1]:.1f}, {prior_angle[2]:.1f})")
        print(f"         Current: ({current_angle[0]:.1f}, {current_angle[1]:.1f}, {current_angle[2]:.1f})")
        print(f"  [MASS] Prior: {has_prior_mass}  |  Current: {has_current_mass}")

        # --- WITH CROP ---
        print("\n  --- Running Variant: WITH CROP ---")
        pair_dir_crop = get_pair_dir(pair_index, input_ct_path, "crop")
        pipeline(
            pair_dir=pair_dir_crop,
            ct_data=ct_data,
            lungs_mask=seg_mask,
            tube_length=tube_length,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            prior_pos=prior_pos,
            current_pos=current_pos,
            prior_angles=prior_angle,
            current_angles=current_angle,
            has_prior_mass=has_prior_mass,
            has_current_mass=has_current_mass,
            use_crop=True,
        )

        # --- WITHOUT CROP ---
        print("\n  --- Running Variant: WITHOUT CROP ---")
        pair_dir_no_crop = get_pair_dir(pair_index, input_ct_path, "no_crop")
        pipeline(
            pair_dir=pair_dir_no_crop,
            ct_data=ct_data,
            lungs_mask=seg_mask,
            tube_length=tube_length,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            prior_pos=prior_pos,
            current_pos=current_pos,
            prior_angles=prior_angle,
            current_angles=current_angle,
            has_prior_mass=has_prior_mass,
            has_current_mass=has_current_mass,
            use_crop=False,
        )

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    end_time = time.time()
    print()
    print("=" * 60)
    print(f" ALL PAIRS COMPLETE!")
    print(f" Outputs saved to: {OUTPUT_DIR}")
    print(f" Total time elapsed: {end_time - start_time:.2f} seconds")
    print("=" * 60)