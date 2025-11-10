import numpy as np
import nibabel as nib
import sys


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Parameters:
        path: Path to .nii or .nii.gz file.
    Returns:
        data: Image data as float32 array.
        affine: 4x4 affine transformation matrix.
        header: Image header with metadata.
    """
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def save_nifti(path: str, data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header) -> None:
    """
    Parameters:
        path: Output file path (.nii or .nii.gz).
        data: Image or label data to save.
        affine: 4x4 affine transformation matrix.
        header: Header to copy metadata from.
    """
    nib.save(nib.Nifti1Image(data, affine, header), path)


def load_and_save_nifti(input_path: str, output_path: str) -> None:
    data, affine, header = load_nifti(input_path)
    save_nifti(output_path, data, affine, header)


def merge_nifti(output_path: str, *input_paths: str) -> None:
    """
    Merge N NIfTI files into a single union mask (non-zero anywhere).
    - Uses the first image as reference for shape/affine/header.
    - Requires all others to match shape and affine.
    """
    if len(input_paths) < 1:
        raise ValueError("Provide at least one input NIfTI path.")

    # Load reference
    data_ref, affine_ref, header_ref = load_nifti(input_paths[0])

    # Start union with first volume's non-zero mask
    union_mask = (data_ref != 0)

    # Merge the rest
    for p in input_paths[1:]:
        data_i, affine_i, header_i = load_nifti(p)

        if data_i.shape != data_ref.shape:
            raise ValueError(f"Shape mismatch for {p}: {data_i.shape} vs {data_ref.shape}")
        elif not np.allclose(affine_i, affine_ref):
            raise ValueError(f"Affine mismatch for {p} (reorient/resample first).")
        elif header_i != header_ref:
            raise ValueError(f"Header mismatch for {p} (reorient/resample first).")

        union_mask |= (data_i != 0)

    save_nifti(output_path, union_mask, affine_ref, header_ref)


if __name__ == "__main__":
    merge_nifti(sys.argv[-1], *sys.argv[1:-1])
