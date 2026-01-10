FILE_EXTENSION = ".nii.gz"

# Option 1- Paths (Itamar's files)
INPUT_DIR = "../itamar_sab/LongitudinalCXRAnalysis/CT_scans/"
OUTPUT_DIR = "./pipeline_output_itamar_files/"
SEG_DIR = "./segmentations_itamar_files/"

# In option 1, comment only one line each time (one folder)
# DATA_FOLDER = "scans/"  # folder 1- scans
# DATA_FOLDER = "LUNA_scans/"  # folder 2- LUNA_scans
DATA_FOLDER = "CT-RATE_scans/"  # folder 3 - CT-RATE_scans folder

INPUT_DIR += DATA_FOLDER
OUTPUT_DIR += DATA_FOLDER
SEG_DIR += DATA_FOLDER

# Option 2- Paths (original 3 ct files)
# INPUT_DIR = "./ct/"
# OUTPUT_DIR = "pipeline_output/"
# SEG_DIR = "./segmentations/"
