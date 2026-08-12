"""
End-to-end pipeline runner.
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

import config as cfg

# --- Global Imports to prevent Multiprocessing Deadlocks ---
from totalsegmentator.python_api import totalsegmentator
from lib.nifti_io import load_nifti, create_seg_path
from datagen.generator import create_pairs_for_scan, _create_output_path
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
    """Stage 1: Generate lung segmentations."""
    print("\n" + "=" * 60)
    print(" STAGE 1: LUNG SEGMENTATION")
    print("=" * 60)

    created = 0
    for filepath in scan_files:
        # Extract just the filename (e.g., train_1031.nii.gz) for safe saving
        filename = os.path.basename(filepath)
        print(f"segmenting: {filename}")

        input_path = filepath # Use the actual location found by os.walk
        seg_path = create_seg_path(filename)

        if os.path.exists(seg_path):
            print(f"  Segmentation exists, skipping: {seg_path}")
            continue

        print(f"  -> Segmenting: {filename} (this may take several minutes)...")

        try:
            totalsegmentator(
                input=input_path, output=seg_path, ml=True,
                task=cfg.SEG_TASK, fast=True, preview=False,
                roi_subset=cfg.ROI_SUBSET,
            )
            created += 1
        except MemoryError:
            print(f"  MemoryError (out of RAM), skipping: {input_path}")
            continue
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    print(f"  Segmentations created: {created}")


def run_data_generation(scan_files):
    """Stage 2: Generate synthetic prior/current/heatmap pairs."""
    print("\n" + "=" * 60)
    print(" STAGE 2: DATA GENERATION")
    print("=" * 60)

    for filepath in scan_files:
        filename = os.path.basename(filepath)
        input_path = filepath
        output_path = _create_output_path(filename)
        seg_path = create_seg_path(filename)

        if os.path.exists(output_path):
            print(f"  Output exists, skipping: {output_path}")
            continue
        elif not os.path.exists(seg_path):
            print(f"  Missing segmentation for {filename}, skipping.")
            continue
        else:
            print(f"  -> Generating pairs for: {filename}...")
            create_pairs_for_scan(input_path, seg_path, filename)


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
    """Functionality for main."""
    parser = argparse.ArgumentParser(description="End-to-end X-ray change detection pipeline.")
    parser.add_argument("ct_dir", type=str, help="Path to the directory containing original CT scans.")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of CT scans to use (0.0–1.0). Default: 1.0 (all).")
    parser.add_argument("--skip-seg", action="store_true",
                        help="Skip the segmentation stage (if already completed).")
    args = parser.parse_args()

    start_time = time.time()

    cfg.set_ct_input_dir(args.ct_dir)

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
