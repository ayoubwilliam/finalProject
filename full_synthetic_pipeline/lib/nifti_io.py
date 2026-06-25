"""
NIfTI file I/O utilities.
Handles loading, saving, merging NIfTI volumes, and 2D image ↔ NIfTI conversion.
"""

import os
import numpy as np
import nibabel as nib

import config as cfg


def load_nifti(path):
    """Loads a .nii/.nii.gz file. Returns (data, affine, header)."""
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def save_nifti(path, data, affine, header):
    """Saves data as a NIfTI file with given affine and header."""
    nib.save(nib.Nifti1Image(data, affine, header), path)


def merge_nifti(output_path, *input_paths):
    """Merges N NIfTI files into a single binary union mask."""
    if len(input_paths) < 1:
        raise ValueError("Provide at least one input NIfTI path.")

    data_ref, affine_ref, header_ref = load_nifti(input_paths[0])
    union_mask = (data_ref != 0)

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


def save_image_as_nifti(data, path):
    """Saves a 2D numpy array as a NIfTI file with identity affine."""
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    generic_affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, generic_affine)
    nib.save(nifti_img, path)


def load_image_from_nifti(path):
    """Loads a NIfTI file and squeezes extra dimensions (for 2D images)."""
    img = nib.load(path)
    data = img.get_fdata()
    data = np.squeeze(data)
    return data


def create_seg_path(filename):
    """Builds the segmentation output path for a given CT filename."""
    os.makedirs(cfg.SEGMENTATION_DIR, exist_ok=True)
    base = filename.split(cfg.NIFTI_EXTENSION)[0]
    return os.path.join(cfg.SEGMENTATION_DIR, base + cfg.SEG_SUFFIX + cfg.NIFTI_EXTENSION)


def create_trachea_seg_path(filename):
    """Builds the trachea-only segmentation output path for a given CT filename."""
    os.makedirs(cfg.SEGMENTATION_DIR, exist_ok=True)
    base = filename.split(cfg.NIFTI_EXTENSION)[0]
    return os.path.join(cfg.SEGMENTATION_DIR, base + cfg.TRACHEA_SEG_SUFFIX + cfg.NIFTI_EXTENSION)
