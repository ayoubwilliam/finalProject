from totalsegmentator.python_api import totalsegmentator
import os
import tempfile

from file_handler import merge_nifti

SEG_TASK = "total"
ROI_SUBSET = ["lung_lower_lobe_right", "lung_upper_lobe_right", "lung_middle_lobe_right",
              "lung_lower_lobe_left", "lung_upper_lobe_left"]

# data paths
INPUT_DIR = "./ct/"
SEG_DIR = "./segmentations/"
CT_FILENAME = "ct_file"
SEG_FILENAME = "lungs"
FILE_EXTENSION = ".nii.gz"

NUMBER_OF_CT_SCANS = 3


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


def create_path(folder: str, filename: str, scan_index: int, is_folder: bool = False) -> str:
    path = folder + filename + str(scan_index)
    return path if is_folder else path + FILE_EXTENSION


def create_lungs_seg() -> None:
    os.makedirs(SEG_DIR, exist_ok=True)

    for scan_index in range(1, NUMBER_OF_CT_SCANS + 1):
        # create paths
        input_path = create_path(INPUT_DIR, CT_FILENAME, scan_index)
        seg_filename = create_path(SEG_DIR, SEG_FILENAME, scan_index)

        # Temporary folder for this scan's lobe outputs
        with tempfile.TemporaryDirectory(prefix=f"totseg_scan{scan_index}_") as tmp_dir:
            run_segmentation(input_path, tmp_dir, SEG_TASK, ROI_SUBSET)
            lobes = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
            merge_nifti(seg_filename, *lobes)


if __name__ == '__main__':
    create_lungs_seg()
