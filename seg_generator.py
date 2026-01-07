from totalsegmentator.python_api import totalsegmentator
import os
import tempfile

from file_handler import merge_nifti, create_seg_path

# data paths
INPUT_DIR = "./ct/"

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
    for filename in os.listdir(INPUT_DIR):
        # create paths
        input_path = os.path.join(INPUT_DIR, filename)
        seg_path = create_seg_path(filename)

        # Temporary folder for this scan's lobe outputs
        with tempfile.TemporaryDirectory(prefix=f"totseg_scan_{filename}_") as tmp_dir:
            run_segmentation(input_path, tmp_dir, SEG_TASK, ROI_SUBSET)
            lobes = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            merge_nifti(seg_path, *lobes)


if __name__ == '__main__':
    create_lungs_seg()
