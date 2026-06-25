"""
Lung segmentation using TotalSegmentator.
Pipeline: For each CT scan → run TotalSegmentator → merge lung lobes → save single mask.
"""

import os
import tempfile

from totalsegmentator.python_api import totalsegmentator

import config as cfg
from lib.nifti_io import merge_nifti, create_seg_path


def run_segmentation(input_path, output_path, task, subset, fast=True):
    """Runs TotalSegmentator on a single CT scan and saves lobe outputs to output_path."""
    totalsegmentator(
        input=input_path,
        output=output_path,
        task=task,
        fast=fast,
        preview=False,
        roi_subset=subset,
    )


def create_lungs_seg():
    """Iterates all CT scans in ct_original, creates merged lung segmentations."""
    total, skipped_exists, created, skipped_perm = 0, 0, 0, 0

    for filename in os.listdir(cfg.CT_ORIGINAL_DIR):
        total += 1

        input_path = os.path.join(cfg.CT_ORIGINAL_DIR, filename)
        seg_path = create_seg_path(filename)

        if os.path.exists(seg_path):
            skipped_exists += 1
            print(f"Segmentation already exists, skipping: {seg_path}")
            continue

        # temporary folder for individual lobe outputs
        with tempfile.TemporaryDirectory(prefix=f"totseg_scan_{filename}_") as tmp_dir:
            try:
                run_segmentation(input_path, tmp_dir, cfg.SEG_TASK, cfg.ROI_SUBSET)
            except PermissionError:
                skipped_perm += 1
                print(f"Permission denied, skipping: {input_path}")
                continue

            lobes = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            merge_nifti(seg_path, *lobes)
            created += 1

    print("total: ", total)
    print("skipped_exists: ", skipped_exists)
    print("skipped_perm: ", skipped_perm)
    print("created: ", created)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run lung segmentation directly.")
    parser.add_argument("ct_dir", type=str, help="Path to the directory containing original CT scans.")
    args = parser.parse_args()
    
    cfg.set_ct_input_dir(args.ct_dir)
    create_lungs_seg()
