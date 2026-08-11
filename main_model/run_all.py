"""
End-to-end pipeline runner for the main model (Consolidation + Tube).
Executes all 4 stages in sequence: Segmentation → Data Generation → Training → Evaluation.
Supports a --data-fraction flag to use a percentage of available CT scans for faster runs.

Usage:
    python run_all.py /path/to/ct/scans            # run full pipeline with all data
    python run_all.py /path/to/ct/scans --data-fraction 0.5 # use 50% of CT scans
    python run_all.py /path/to/ct/scans --skip-seg          # skip segmentation (already done)
"""

import argparse
import os
import time
import random
import tempfile
import shutil

import config as cfg

# --- Global Imports to prevent Multiprocessing Deadlocks ---
from totalsegmentator.python_api import totalsegmentator
from lib.nifti_io import load_nifti, merge_nifti, create_lungs_seg_path
from datagen.generator import create_pairs_for_scan, _create_output_path, create_trachea_seg_path
from training.trainer import train
from evaluation.evaluate import main as eval_main


def _get_scan_subset(fraction):
    """Returns a subset of CT absolute file paths based on the requested fraction."""
    all_files = []

    # RECURSIVE SEARCH: Dive into all subdirectories to find .nii or .nii.gz files
    for root, dirs, files in os.walk(cfg.CT_ORIGINAL_DIR):
        for f in files:
            if f.endswith(('.nii.gz', '.nii')):
                # Save the full absolute path to the file
                all_files.append(os.path.join(root, f))

    all_files = sorted(all_files)

    if not all_files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found anywhere inside {cfg.CT_ORIGINAL_DIR} or its subdirectories.")

    if fraction >= 1.0:
        return all_files

    n = max(1, int(len(all_files) * fraction))
    random.seed(cfg.SEED)
    subset = random.sample(all_files, n)
    print(f"[INFO] Using {len(subset)}/{len(all_files)} scans ({fraction * 100:.0f}%)")
    return subset


def run_segmentation(scan_files):
    """Stage 1: Generate required segmentations (lungs and/or trachea)."""
    print("\n" + "=" * 60)
    print(" STAGE 1: SEGMENTATION")
    print("=" * 60)

    # Force temporary files to be written to the DATA_DIR drive to prevent C: drive exhaustion
    custom_tmp_dir = os.path.join(cfg.DATA_DIR, "tmp_totalseg")
    os.makedirs(custom_tmp_dir, exist_ok=True)

    created_lungs = 0
    created_trachea = 0
    
    for filepath in scan_files:
        filename = os.path.basename(filepath)
        print(f"segmenting: {filename}")

        input_path = filepath
        lungs_seg_path = create_lungs_seg_path(filename)
        trachea_seg_path = create_trachea_seg_path(filename)

        lungs_exists = os.path.exists(lungs_seg_path)
        trachea_exists = os.path.exists(trachea_seg_path)

        needs_lungs = cfg.ADD_CONSOLIDATION and not lungs_exists
        needs_trachea = cfg.ADD_TUBE and not trachea_exists

        if not needs_lungs and not needs_trachea:
            print(f"  Segmentations exist or are not required, skipping: {filename}")
            continue

        print(f"  -> Segmenting: {filename} (this may take several minutes)...")

        # Determine which subsets to run
        current_roi_subset = []
        if needs_lungs:
            current_roi_subset.extend([
                "lung_lower_lobe_right", "lung_upper_lobe_right", "lung_middle_lobe_right",
                "lung_lower_lobe_left", "lung_upper_lobe_left"
            ])
        if needs_trachea:
            current_roi_subset.append("trachea")

        with tempfile.TemporaryDirectory(prefix=f"totseg_{filename}_", dir=custom_tmp_dir) as tmp_dir:
            try:
                totalsegmentator(
                    input=input_path, output=tmp_dir,
                    task=cfg.SEG_TASK, fast=True, preview=False,
                    roi_subset=current_roi_subset,
                )
            except PermissionError:
                print(f"  Permission denied, skipping: {input_path}")
                continue

            # Merge and save lungs
            if needs_lungs:
                lung_lobes = [
                    os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) 
                    if "lung" in f and f.endswith(".nii.gz")
                ]
                if lung_lobes:
                    merge_nifti(lungs_seg_path, *lung_lobes)
                    created_lungs += 1

            # Move and save trachea
            if needs_trachea:
                trachea_file = os.path.join(tmp_dir, "trachea.nii.gz")
                if os.path.exists(trachea_file):
                    shutil.move(trachea_file, trachea_seg_path)
                    created_trachea += 1

    if os.path.exists(custom_tmp_dir):
        try:
            shutil.rmtree(custom_tmp_dir)
        except OSError:
            pass

    print(f"  Lungs segmentations created: {created_lungs}")
    print(f"  Trachea segmentations created: {created_trachea}")


def run_data_generation(scan_files):
    """Stage 2: Generate synthetic prior/current/heatmap pairs."""
    print("\n" + "=" * 60)
    print(" STAGE 2: DATA GENERATION")
    print("=" * 60)

    for filepath in scan_files:
        filename = os.path.basename(filepath)
        input_path = filepath
        output_path = _create_output_path(filename)
        
        lungs_seg_path = create_lungs_seg_path(filename)
        trachea_seg_path = create_trachea_seg_path(filename)

        if os.path.exists(output_path):
            print(f"  Output exists, skipping: {output_path}")
            continue
            
        if cfg.ADD_CONSOLIDATION and not os.path.exists(lungs_seg_path):
            print(f"  Missing lungs segmentation for {filename}, skipping.")
            continue
            
        if cfg.ADD_TUBE and not os.path.exists(trachea_seg_path):
            print(f"  Missing trachea segmentation for {filename}, skipping.")
            continue

        print(f"  -> Generating pairs for: {filename}...")
        create_pairs_for_scan(input_path, lungs_seg_path, trachea_seg_path, filename)


def run_training():
    """Stage 3: Train the model."""
    print("\n" + "=" * 60)
    print(" STAGE 3: TRAINING")
    print("=" * 60)

    train()


def run_evaluation():
    """Stage 4: Run full evaluation."""
    print("\n" + "=" * 60)
    print(" STAGE 4: EVALUATION")
    print("=" * 60)

    eval_main()


def main():
    parser = argparse.ArgumentParser(description="End-to-end X-ray change detection pipeline for main model.")
    parser.add_argument("--ct_dir", type=str, default="", help="Path to the directory containing original CT scans.")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of CT scans to use (0.0–1.0). Default: 1.0 (all).")
    parser.add_argument("--skip-seg", action="store_true",
                        help="Skip the segmentation stage (if already completed).")
    args = parser.parse_args()

    start_time = time.time()
    
    # Check if a directory was provided. Demos might just run the pipeline on the internal directory.
    if args.ct_dir:
        cfg.set_ct_input_dir(args.ct_dir)
    elif cfg.CT_ORIGINAL_DIR is None:
        # Fallback for demos running without args
        cfg.set_ct_input_dir(os.path.join(cfg.DATA_DIR, "sample_cts"))

    try:
        scan_files = _get_scan_subset(args.data_fraction)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    if not args.skip_seg:
        run_segmentation(scan_files)
    else:
        print("[INFO] Skipping segmentation stage.")

    run_data_generation(scan_files)
    run_training()
    run_evaluation()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" ALL STAGES COMPLETE! Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
