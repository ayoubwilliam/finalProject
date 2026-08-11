"""
Dataset and DataLoader factories for change detection.
Loads prior/current/heatmap NIfTI triplets, splits at the scan level to prevent data leakage.
Shared between training and evaluation.
"""

import os
import glob
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config as cfg


# ==============================================================================
# DATASET DEFINITION
# ==============================================================================

class ChangeDetectionDataset(Dataset):
    """Loads Prior, Current, and Target NIfTI images as PyTorch tensors."""

    def __init__(self, root_dir, file_list):
        self.root_dir = root_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pair_path = os.path.join(self.root_dir, self.file_list[idx])

        try:
            prior_np, current_np, heatmap_np = self._load_nifti_triplet(pair_path)
            
            # Load true angles
            import json
            with open(os.path.join(pair_path, "angles.json"), "r") as f:
                angles_dict = json.load(f)
            
            prior_angles = angles_dict["prior_angles"]
            current_angles = angles_dict["current_angles"]
            
            # Assuming projection axis is 1 (Y-axis), so angle[1] is the in-plane rotation
            true_angle = current_angles[1] - prior_angles[1]
            
            return (
                self._to_tensor(prior_np),
                self._to_tensor(current_np),
                torch.tensor(true_angle, dtype=torch.float32),
                self.file_list[idx],
                self._to_tensor(heatmap_np),
            )
        except Exception as e:
            print(f"\n[WARNING] Corrupt file detected at {pair_path}: {e}")
            random_idx = random.randint(0, len(self.file_list) - 1)
            return self.__getitem__(random_idx)

    def _load_nifti_triplet(self, pair_path):
        """Loads the three core NIfTI matrices from a pair directory."""
        prior = nib.load(os.path.join(pair_path, "prior.nii.gz")).get_fdata()
        current = nib.load(os.path.join(pair_path, "current.nii.gz")).get_fdata()
        heatmap = nib.load(os.path.join(pair_path, "heatmap.nii.gz")).get_fdata()
        return prior, current, heatmap

    def _to_tensor(self, img):
        """Formats a numpy array into a PyTorch tensor with channel dimension."""
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            img = np.moveaxis(img, -1, 0)
        return torch.from_numpy(img).float()


# ==============================================================================
# DATA SPLITTING
# ==============================================================================

def _group_valid_scans():
    """Finds all valid Pair directories and groups them by parent scan ID."""
    search_path = os.path.join(cfg.GENERATED_SYNTHETIC_DIR, "**", "Pair*")
    all_pairs = glob.glob(search_path, recursive=True)

    scan_to_pairs = {}
    required_files = ("prior.nii.gz", "current.nii.gz", "heatmap.nii.gz", "angles.json")

    for pair_path in all_pairs:
        has_all_files = all(os.path.exists(os.path.join(pair_path, f)) for f in required_files)

        if has_all_files:
            rel_path = os.path.relpath(pair_path, cfg.GENERATED_SYNTHETIC_DIR)
            scan_id = os.path.dirname(rel_path)
            scan_to_pairs.setdefault(scan_id, []).append(rel_path)

    if not scan_to_pairs:
        raise RuntimeError("No valid data pairs found in the specified directory!")

    return scan_to_pairs


def get_data_split():
    """Creates a train/test split at the scan level to prevent data leakage."""
    scan_to_pairs = _group_valid_scans()

    scan_ids = sorted(scan_to_pairs.keys())
    train_scans, test_scans = train_test_split(
        scan_ids,
        test_size=cfg.VAL_SPLIT,
        random_state=cfg.SEED,
    )

    train_list = [p for scan in train_scans for p in scan_to_pairs[scan]]
    test_list = [p for scan in test_scans for p in scan_to_pairs[scan]]

    return train_list, test_list


# ==============================================================================
# DATALOADER FACTORIES
# ==============================================================================

def get_train_dataloader():
    """Builds the training DataLoader (shuffled)."""
    train_files, _ = get_data_split()
    ds = ChangeDetectionDataset(cfg.GENERATED_SYNTHETIC_DIR, train_files)

    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )


def get_test_dataloader():
    """Builds the evaluation DataLoader (unshuffled) and returns the test file list."""
    _, test_files = get_data_split()
    ds = ChangeDetectionDataset(cfg.GENERATED_SYNTHETIC_DIR, test_files)

    loader = DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return loader, test_files
