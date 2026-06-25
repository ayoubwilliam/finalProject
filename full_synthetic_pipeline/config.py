"""
Universal configuration for the entire X-ray change detection project.
All paths, magic numbers, hyperparameters, and tunable constants live here.
"""

import os
import tempfile
import torch
import numpy as np

# ==============================================================================
# PROJECT ROOT, DATA PATHS & TEMP SETUP
# ==============================================================================

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), "data")

# --- FIX FOR [Errno 28] No space left on device ---
# Create a custom temporary directory inside your lab storage space
CUSTOM_TMP_DIR = os.path.join(DATA_DIR, "tmp")
os.makedirs(CUSTOM_TMP_DIR, exist_ok=True)

# Force Python and external libraries to use this directory instead of the system /tmp
tempfile.tempdir = CUSTOM_TMP_DIR
os.environ['TMPDIR'] = CUSTOM_TMP_DIR
os.environ['TEMP'] = CUSTOM_TMP_DIR
os.environ['TMP'] = CUSTOM_TMP_DIR
# --------------------------------------------------

CT_ORIGINAL_DIR = None  # set at runtime via set_ct_input_dir()
SEGMENTATION_DIR = os.path.join(DATA_DIR, "segmentations")
GENERATED_SYNTHETIC_DIR = os.path.join(DATA_DIR, "generated_synthetic")
MODEL_WEIGHTS_DIR = os.path.join(DATA_DIR, "model_weights")
EVALUATION_DIR = os.path.join(DATA_DIR, "evaluation")

# Evaluation sub-folders
EVAL_VISUALS_DIR = os.path.join(EVALUATION_DIR, "Visuals")
EVAL_PW_DIR = os.path.join(EVALUATION_DIR, "PW")
EVAL_CC_DIR = os.path.join(EVALUATION_DIR, "CC")
EVAL_CC_VISUALS_DIR = os.path.join(EVAL_CC_DIR, "Visual_Samples")
EVAL_CC_METRICS_DIR = os.path.join(EVAL_CC_DIR, "Dataset_Metrics")

# Evaluation output files
EVAL_FILE_PW_DICE = os.path.join(EVAL_PW_DIR, "pw_dice_report.txt")
EVAL_FILE_PW_ROC = os.path.join(EVAL_PW_DIR, "pw_roc_curve.png")
EVAL_FILE_CC_METRICS = os.path.join(EVAL_CC_METRICS_DIR, "cc_object_evaluation.txt")
EVAL_FILE_CC_DICE = os.path.join(EVAL_CC_METRICS_DIR, "cc_instance_dice.txt")

# ==============================================================================
# FILE CONVENTIONS
# ==============================================================================

NIFTI_EXTENSION = ".nii.gz"
SEG_SUFFIX = "_lungs_seg"
TRACHEA_SEG_SUFFIX = "_trachea_seg"

# Image filenames written by the data generation pipeline
CURRENT_FILENAME = "current.png"
PRIOR_BY_PRIOR_FILENAME = "prior_rotated_to_prior.png"
PRIOR_BY_CURRENT_FILENAME = "prior_rotated_to_current.png"
HEATMAP_FILENAME = "heatmap.png"

# ==============================================================================
# DEVICE
# ==============================================================================

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

# ==============================================================================
# SEGMENTATION
# ==============================================================================

SEG_TASK = "total"
ROI_SUBSET = [
    "lung_lower_lobe_right", "lung_upper_lobe_right", "lung_middle_lobe_right",
    "lung_lower_lobe_left", "lung_upper_lobe_left",
    "trachea",
]

# ==============================================================================
# DATA GENERATION — ET-TUBE GEOMETRY
# ==============================================================================

# Outer wall radius in voxels (the visible radiopaque shell)
TUBE_OUTER_RADIUS_MIN = 3
TUBE_OUTER_RADIUS_MAX = 5

# Inner lumen radius in voxels (the air-filled center channel)
TUBE_INNER_RADIUS_MIN = 1
TUBE_INNER_RADIUS_MAX = 3

# Tube length in voxels along its primary axis
TUBE_LENGTH_MIN = 100
TUBE_LENGTH_MAX = 200

# HU / intensity values
TUBE_WALL_INTENSITY = 2000   # Radiopaque tube wall — very bright on DRR
TUBE_LUMEN_INTENSITY = -900  # Air-filled lumen — dark on X-ray

# ==============================================================================
# DATA GENERATION — ET-TUBE DEFORMATION (B-SPLINE)
# ==============================================================================

# Coarser grid → fewer control points → lower-frequency (smoother) bends
TUBE_GRID_DENSITY_FACTOR = 4

# Very small displacement → preserves tubular topology, avoids collapse
TUBE_DEFORMATION_FACTOR = 0.03

# ==============================================================================
# DATA GENERATION — BOUNDARY SMOOTHING
# ==============================================================================

# Pooling kernel for tube boundary smoothing (smaller than sphere's 8
# because tube walls are only 3-5 voxels thick)
TUBE_POOLING_KERNEL_SIZE = 4

# ==============================================================================
# DATA GENERATION — NOISE (applied to tube wall texture)
# ==============================================================================

NOISE_MEAN = 0
NOISE_STD = 20
NOISE_BLOCK_FACTOR = 4

# ==============================================================================
# DATA GENERATION — RANDOMIZATION
# ==============================================================================

NUMBER_OF_PAIRS_PER_SCAN = 5
ROT_ANGLE_X_RANGE_DEG = 10.0
ROT_ANGLE_Y_RANGE_DEG = 5.0
ROT_ANGLE_Z_RANGE_DEG = 5.0
ADD_MASS_PRIOR_PROBABILITY = 1.0
ADD_MASS_CURRENT_PROBABILITY = 1.0

# ==============================================================================
# DRR — PRE / POST PROCESSING
# ==============================================================================

AIR_CT_THRESHOLD = -1000
CT_CLIP_MAX = 3000
DRR_TARGET_SIZE = (512, 512)
DRR_PROJECTION_AXIS = 1
CROP_MARGIN = 10

CLAHE_WINDOW_SIZE = 4
CLAHE_CLIP_LIMIT = 5.0
SHARPNESS_FACTOR = 3.0

# ==============================================================================
# MODEL & TRAINING
# ==============================================================================

SELECTED_MODEL = "VGGDiffNet"  # Options: "SimpleDiffNet", "VGGDiffNet"
MODEL_WEIGHTS_FILENAME = "trained_model_weights.pth"
MODEL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, MODEL_WEIGHTS_FILENAME)

SEED = 42
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
VAL_SPLIT = 0.2
NUM_WORKERS = 4
PIN_MEMORY = True

# ==============================================================================
# EVALUATION THRESHOLDS
# ==============================================================================

PW_DICE_THRESHOLD = 0.03
PRED_THRESHOLD = 0.03
GT_THRESHOLD = 0.03
IOU_THRESHOLD = 0.3
CC_VISUAL_MAX_SAMPLES = 60
MATH_EPSILON = 1e-5
MIN_CC_SIZE = 50

# ==============================================================================
# PLOTTING & VISUALIZATION
# ==============================================================================

PLOT_DPI_STANDARD = 100
PLOT_DPI_HIGH = 300
PLOT_FIGSIZE_STANDARD = (18, 4)
PLOT_FIGSIZE_ROC = (10, 8)
PLOT_FIGSIZE_CC = (26, 30)
NIFTI_AFFINE = np.eye(4)

# ==============================================================================
# AUTO-CREATE DATA DIRECTORIES
# ==============================================================================


def ensure_directories():
    """Creates all required data directories if they don't already exist."""
    for d in [
        SEGMENTATION_DIR, GENERATED_SYNTHETIC_DIR, MODEL_WEIGHTS_DIR,
        EVALUATION_DIR, EVAL_VISUALS_DIR, EVAL_PW_DIR,
        EVAL_CC_DIR, EVAL_CC_VISUALS_DIR, EVAL_CC_METRICS_DIR,
        CUSTOM_TMP_DIR # Also ensure our new tmp directory is included
    ]:
        os.makedirs(d, exist_ok=True)


def set_ct_input_dir(path):
    """Sets the CT input directory at runtime. Must be called before seg/datagen."""
    global CT_ORIGINAL_DIR
    CT_ORIGINAL_DIR = os.path.abspath(path)
    if not os.path.isdir(CT_ORIGINAL_DIR):
        raise FileNotFoundError(f"CT input directory not found: {CT_ORIGINAL_DIR}")
    print(f"[CONFIG] CT input directory: {CT_ORIGINAL_DIR}")


ensure_directories()