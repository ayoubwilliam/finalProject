import torch

# ==============================================================================
# HARDWARE
# ==============================================================================
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

# ==============================================================================
# DRR — PRE / POST PROCESSING
# ==============================================================================
# CT Hounsfield Unit clipping thresholds
AIR_CT_THRESHOLD = -1000.0
CT_CLIP_MAX = 3000.0

# 0 = Sagittal, 1 = Coronal (AP), 2 = Axial
DRR_PROJECTION_AXIS = 1

# Final output resolution for the DRR
DRR_TARGET_SIZE = (512, 512)

# Post-processing enhancement parameters
CLAHE_WINDOW_SIZE = 4
CLAHE_CLIP_LIMIT = 5.0
SHARPNESS_FACTOR = 3.0

# ==============================================================================
# TUBE GENERATION CONSTANTS
# ==============================================================================
PATH_SMOOTHING_SIGMA = 15.0
SHAVING_SIGMA = 2.5

# ==============================================================================
# PIPELINE CONSTANTS
# ==============================================================================
CURRENT_FILENAME = "current.png"
PRIOR_BY_PRIOR_FILENAME = "prior.png"
PRIOR_BY_CURRENT_FILENAME = "prior_by_current.png"
HEATMAP_FILENAME = "heatmap.png"
GT_THRESHOLD = 1e-4

# ==============================================================================
# DATA GENERATION RANDOMIZATION & CROP
# ==============================================================================
ROT_ANGLE_X_RANGE_DEG = 10.0
ROT_ANGLE_Y_RANGE_DEG = 5.0
ROT_ANGLE_Z_RANGE_DEG = 5.0
ADD_MASS_PRIOR_PROBABILITY = 1
ADD_MASS_CURRENT_PROBABILITY = 1
CROP_MARGIN = 10