import os
import numpy as np
import nibabel as nib
import tempfile
from totalsegmentator.python_api import totalsegmentator

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_CT_PATH = "../ct/ct_file1.nii.gz"
SEGS_DIR = "./segs"

# Define the highly specific tasks, their ROIs, and whether they support the 'fast' flag
AIRWAY_TASKS = [
    {
        "task": "total",
        "subset": ["trachea"],
        "fast": True  # CAN run fast to save time
    }
]


# ==========================================
# 2. MULTI-TASK SEGMENTATION ENGINE
# ==========================================
def generate_full_airway_mask(ct_path, segs_dir):
    """
    Runs TotalSegmentator across multiple specific sub-tasks,
    collects the targeted ROIs, and merges them into one master airway mask.
    """
    os.makedirs(segs_dir, exist_ok=True)
    base_name = os.path.basename(ct_path).replace(".nii.gz", "").replace(".nii", "")
    final_save_path = os.path.join(segs_dir, f"{base_name}_full_airway_v2.nii.gz")

    # 1. Check Cache
    if os.path.exists(final_save_path):
        print(f"Found cached full airway segmentation at {final_save_path}. Skipping.")
        return final_save_path

    print(f"Starting multi-task airway extraction for {base_name}...")

    # Load original CT to get the exact shape and affine matrix for the blank canvas
    ct_img = nib.load(ct_path)
    merged_mask = np.zeros(ct_img.shape, dtype=np.uint8)
    found_any_rois = False

    # 2. Iterate through the tasks
    with tempfile.TemporaryDirectory() as tmp_dir:
        for job in AIRWAY_TASKS:
            current_task = job["task"]
            current_rois = job["subset"]
            use_fast = job["fast"]

            print(f"\n--- Running Task: '{current_task}' for ROIs: {current_rois} (fast={use_fast}) ---")

            task_dir = os.path.join(tmp_dir, current_task)
            os.makedirs(task_dir, exist_ok=True)

            try:
                totalsegmentator(
                    input=ct_path,
                    output=task_dir,
                    task=current_task,
                    fast=use_fast,
                    preview=False,
                    roi_subset=current_rois
                )

                # Merge the outputs from this specific task into our master canvas
                for roi in current_rois:
                    roi_file = os.path.join(task_dir, f"{roi}.nii.gz")
                    if os.path.exists(roi_file):
                        roi_data = nib.load(roi_file).get_fdata()
                        merged_mask[roi_data > 0] = 1
                        found_any_rois = True
                        print(f"  -> Successfully extracted and merged: {roi}")
                    else:
                        print(f"  -> Warning: Could not find '{roi}' (Anatomy might be cut off in this CT scan).")

            except Exception as e:
                print(f"  -> Error during task '{current_task}': {e}")
                continue

    # 3. Final Validation & Save
    if not found_any_rois:
        raise ValueError(f"TotalSegmentator failed to find ANY airway ROIs in {ct_path}!")

    print(f"\nSaving master airway segmentation to {final_save_path}...")
    out_img = nib.Nifti1Image(merged_mask, ct_img.affine, ct_img.header)
    nib.save(out_img, final_save_path)

    return final_save_path


if __name__ == "__main__":
    os.environ["PyTorch_GEOM_ALLOCATOR"] = "CUDA_MALLOC"
    generate_full_airway_mask(INPUT_CT_PATH, SEGS_DIR)