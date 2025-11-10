import sys
import os
from totalsegmentator.python_api import totalsegmentator


def run_segmentation(input_path: str, base_output_dir: str, task: str, subset: list,
                     fast: bool = True) -> None:
    """
    Run TotalSegmentator and save results into a dedicated folder for this task.
    """
    # Create a subdirectory for the specific task
    output_folder = os.path.join(base_output_dir, subset[0] if subset else task)
    os.makedirs(output_folder, exist_ok=True)

    totalsegmentator(
        input=input_path,
        output=output_folder,
        task=task,
        fast=fast,
        preview=False,
        roi_subset=subset
    )


if __name__ == "__main__":
    input_path = sys.argv[1]
    base_output_dir = sys.argv[2]
    task = sys.argv[3]
    subset = sys.argv[4:]
    run_segmentation(input_path, base_output_dir, task, subset)
