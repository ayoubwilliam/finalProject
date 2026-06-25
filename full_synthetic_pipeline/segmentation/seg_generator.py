"""
Lung + airway segmentation using TotalSegmentator.
Pipeline: For each CT scan → run TotalSegmentator → merge lung lobes + trachea → save merged mask
          AND also save the trachea mask separately for tube placement.
"""

import os
import tempfile
import shutil

from totalsegmentator.python_api import totalsegmentator

import config as cfg
from lib.nifti_io import merge_nifti, save_nifti, load_nifti, create_seg_path, create_trachea_seg_path


def run_segmentation(input_path, output_path, task, subset, fast=True):
    """Runs TotalSegmentator on a single CT scan and saves lobe outputs to output_path."""
    try:
        totalsegmentator(
            input=input_path,
            output=output_path,
            task=task,
            fast=fast,
            preview=False,
            roi_subset=subset,
        )
    except Exception as e:
        if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
            print(f"[SEG] Warning: TotalSegmentator threw {type(e).__name__} during cleanup. Ignoring since outputs exist.")
        else:
            import traceback
            traceback.print_exc()
            raise


def _save_trachea_mask(tmp_dir, trachea_path):
    """
    Finds the trachea.nii.gz file in the TotalSegmentator output directory
    and copies it to a permanent location as the trachea-only mask.
    """
    trachea_file = os.path.join(tmp_dir, "trachea.nii.gz")
    if os.path.exists(trachea_file):
        shutil.copy2(trachea_file, trachea_path)
        print(f"  Trachea mask saved: {trachea_path}")
    else:
        # Fallback: if trachea wasn't segmented, create an empty mask
        # from the first available file's shape
        nii_files = [f for f in os.listdir(tmp_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        if nii_files:
            import numpy as np
            data, affine, header = load_nifti(os.path.join(tmp_dir, nii_files[0]))
            empty = np.zeros_like(data)
            save_nifti(trachea_path, empty, affine, header)
            print(f"  WARNING: trachea.nii.gz not found in TotalSegmentator output. Saved empty trachea mask.")


def create_lungs_seg():
    """Iterates all CT scans in ct_original, creates merged lung+airway segmentations."""
    total, skipped_exists, created, skipped_perm = 0, 0, 0, 0

    for filename in os.listdir(cfg.CT_ORIGINAL_DIR):
        total += 1

        input_path = os.path.join(cfg.CT_ORIGINAL_DIR, filename)
        seg_path = create_seg_path(filename)
        trachea_path = create_trachea_seg_path(filename)

        if os.path.exists(seg_path) and os.path.exists(trachea_path):
            skipped_exists += 1
            print(f"Segmentation already exists, skipping: {seg_path}")
            continue

        # temporary folder for individual lobe outputs
        tmp_dir = tempfile.mkdtemp(prefix=f"totseg_scan_{filename}_")
        try:
            try:
                run_segmentation(input_path, tmp_dir, cfg.SEG_TASK, cfg.ROI_SUBSET)
            except Exception as e:
                skipped_perm += 1
                print(f"Error during segmentation, skipping: {e}")
                continue

            # Save trachea mask separately BEFORE merging/cleanup
            _save_trachea_mask(tmp_dir, trachea_path)

            lobes = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
            if not lobes:
                continue
            merge_nifti(seg_path, *lobes)
            created += 1
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("total: ", total)
    print("skipped_exists: ", skipped_exists)
    print("skipped_perm: ", skipped_perm)
    print("created: ", created)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run lung+airway segmentation directly.")
    parser.add_argument("ct_dir", type=str, help="Path to the directory containing original CT scans.")
    args = parser.parse_args()
    
    cfg.set_ct_input_dir(args.ct_dir)
    create_lungs_seg()
