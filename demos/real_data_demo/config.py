"""
Universal configuration for the X-ray change detection project (Inference).
"""

import torch

# ==============================================================================
# DEVICE
# ==============================================================================
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

# ==============================================================================
# CROP CONSTANTS
# ==============================================================================
# Consolidation crop
CONSOLIDATION_CROP_SETTINGS = [
    (0, 1.00),
    (20, 0.96),
    (40, 0.94),
    (60, 0.92),
    (80, 0.90),
]
CONSOLIDATION_BASE_CROP = (0, 1)

# Tube crop
TUBE_CROP_SETTINGS = [
    (0, 0.80),
    (0, 0.75),
    (0, 0.70),
    (0, 0.65),
    (0, 0.60)
]
TUBE_BASE_CROP = (0, 0.75)

# Translation augmentation (relative offsets: -1.0 for top/left, 0.0 for center, 1.0 for bottom/right)
TRANSLATION_SETTINGS = [
    (0.0, 0.0),    # Center
    # (-1.0, -1.0),  # Top-Left
    # (-1.0, 1.0),   # Top-Right
    # (1.0, -1.0),   # Bottom-Left
    # (1.0, 1.0),    # Bottom-Right
    # (-1.0, 0.0),   # Top-Center
    # (1.0, 0.0),    # Bottom-Center
    # (0.0, -1.0),   # Middle-Left
    # (0.0, 1.0),    # Middle-Right
]
BASE_TRANSLATION = (0.0, 0.0)

# ==============================================================================
# DRR — PRE / POST PROCESSING
# ==============================================================================
DRR_TARGET_SIZE = (512, 512)
CLAHE_WINDOW_SIZE = 4
CLAHE_CLIP_LIMIT = 5.0
SHARPNESS_FACTOR = 3.0

# ==============================================================================
# INFERENCE & VISUALIZATION
# ==============================================================================
PRED_THRESHOLD = 0.015
SELECTED_MODEL = "VGGDiffNet"  # Options: "SimpleDiffNet", "VGGDiffNet"