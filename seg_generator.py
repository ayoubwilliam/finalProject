from totalsegmentator.python_api import totalsegmentator
import os
import tempfile

from file_handler import merge_nifti, create_seg_path
from project_paths import INPUT_DIR

# segmentation constants
SEG_TASK = "total"
ROI_SUBSET = ["lung_lower_lobe_right", "lung_upper_lobe_right", "lung_middle_lobe_right",
              "lung_lower_lobe_left", "lung_upper_lobe_left"]


def run_segmentation(input_path: str, output_path: str, task: str, subset: list,
                     fast: bool = True) -> None:
    """
    Run TotalSegmentator and save results into a dedicated folder for this task.
    """
    totalsegmentator(
        input=input_path,
        output=output_path,
        task=task,
        fast=fast,
        preview=False,
        roi_subset=subset
    )


def create_lungs_seg() -> None:
    total, skipped_exists, created, skipped_perm = 0, 0, 0, 0

    for filename in os.listdir(INPUT_DIR):
        total += 1

        # create paths
        input_path = os.path.join(INPUT_DIR, filename)
        seg_path = create_seg_path(filename)

        if os.path.exists(seg_path):
            skipped_exists += 1
            print(f"Segmentation already exists, skipping: {seg_path}")
            continue

        # Temporary folder for this scan's lobe outputs
        with tempfile.TemporaryDirectory(prefix=f"totseg_scan_{filename}_") as tmp_dir:
            try:
                run_segmentation(input_path, tmp_dir, SEG_TASK, ROI_SUBSET)
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


if __name__ == '__main__':
    create_lungs_seg()
