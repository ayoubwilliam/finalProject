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