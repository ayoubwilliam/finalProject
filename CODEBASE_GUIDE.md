# 📂 Codebase Structure Guide

This guide provides an exhaustive, one-to-one mapping of every file and directory present in the main GitHub repository. The project is modularized into four primary domains: the core twin-encoder model (`main_model`), a standard classification comparator (`baseline_model`), clinical demonstration orchestrators (`demos`), and dataset acquisition tools (`data/real_xray`).

---

## 1. 🧠 Main Model (`main_model/`)
This is the core directory of the project, containing the full End-to-End procedural data generation, PyTorch training, and Connected Components evaluation pipeline for the **VGGDiffNet** twin-encoder architecture.

### Root Level
- **`config.py`**: The universal configuration file defining absolute paths, synthesis hyperparameters, model learning rates, and tunable constants.
- **`file_handler.py`**: Utility library for safely parsing, validating, loading, and saving multi-dimensional medical imaging files.
- **`project_paths.py`**: A helper that standardizes directory navigation and path resolution relative to the repository root.
- **`run_all.py`**: The master orchestrator script that chains Data Generation, Training, and Evaluation sequentially.
- **`run_datagen.py`**: Standalone executor to trigger only the 3D-to-2D data synthesis pipeline.
- **`run_training.py`**: Standalone executor to trigger only the PyTorch neural network optimization loop.
- **`run_evaluation.py`**: Standalone executor to evaluate the trained model on unseen testing data.
- **`seg_generator.py`**: Invokes the TotalSegmentator library to automatically generate essential organ bounding segmentations from raw 3D CT scans.

### `datagen/` (Procedural Synthesis)
- **`__init__.py`**: Python package initializer.
- **`crop.py`**: Optimizes synthesis by applying tight bounding-box crops to isolate the lungs in 3D.
- **`deformation.py`**: Generates B-spline transformations to warp baseline spheres into organic, irregular shapes.
- **`drr.py`**: Applies Digitally Reconstructed Radiography, collapsing 3D CT volumes into 2D synthetic X-rays.
- **`generator.py`**: Iterates over the raw dataset, manages patient files, and delegates synthesis tasks.
- **`noise.py`**: Injects statistical artifacts and Gaussian noise to simulate real-world X-ray sensor degradation.
- **`pipeline.py`**: The primary data generation logic that sequentially processes a single 3D CT pair and manages memory.
- **`pooling.py`**: Applies 3D volume pooling to smooth and blend the generated 3D anomalies.
- **`rotation.py`**: Applies 3D volume rotations independently to prior and current scans to simulate patient posture shifts.
- **`shapes.py`**: Core mathematical functions for generating base 3D spherical consolidations.
- **`tube.py`**: Core logic for generating synthetic Endotracheal (ET) tubes in 3D space.
- **`tube_randomization.py`**: Positions and curves the ET tubes using Euclidean Distance Transforms (EDT) and random walks within the trachea.

### `evaluation/` (Testing & Metrics)
- **`__init__.py`**: Python package initializer.
- **`cc_dice.py`**: Calculates standard object-level Dice coefficients to measure spatial prediction overlap.
- **`cc_lib.py`**: Implements the core mathematical algorithms for Object-Level Connected Components (CC) analysis.
- **`cc_metrics.py`**: Calculates spatial Intersection-over-Union (IoU) and bounding box overlaps for accurate True Positive/False Negative spatial scoring.
- **`cc_visuals.py`**: Generates visual representations of the connected components for spatial debugging.
- **`evaluate.py`**: The top-level orchestrator that evaluates the network on the test dataset.
- **`visuals.py`**: Generates intuitive PNG comparison grids showing the Prior X-ray, Current X-ray, Ground Truth, and the AI's Predicted Heatmap side-by-side.

### `lib/` (Helpers)
- **`__init__.py`**: Python package initializer.
- **`nifti_io.py`**: Highly optimized helper functions for reading and writing `.nii.gz` arrays to disk.
- **`overlay_heatmap.py`**: Applies colormaps (e.g., JET) to grayscale model outputs to create visual anomaly overlays.

### `training/` (Deep Learning)
- **`__init__.py`**: Python package initializer.
- **`data_loader.py`**: Custom PyTorch Dataset classes for batched fetching of generated 2D X-ray image pairs.
- **`model_io.py`**: Safely manages saving, loading, and checkpointing neural network `.pth` weight files.
- **`models.py`**: Defines the `VGGDiffNet` architecture, the weight-sharing twin-encoders, and the feature-subtraction bottleneck.
- **`trainer.py`**: The core PyTorch training loop, including loss calculations, backpropagation, and mixed-precision scaling.

---

## 2. ⚖️ Baseline Model (`baseline_model/`)
This directory contains a standard classifier architecture used as a baseline comparator to prove the efficacy of the twin-encoder VGGDiffNet. It has a parallel but simplified structure.

### Root Level
- **`config.py`**: Configurations specifically tailored for the baseline model's training and data sizes.
- **`run_all.py`**: Orchestrates data generation and training for the baseline classifier.
- **`run_pipeline_on_single_test.py`**: Utility to test a single image through the trained baseline model.

### `datagen/` (Baseline Synthesis)
- **`__init__.py`**: Python package initializer.
- **`drr.py`**: Applies DRR tailored for generating single (non-paired) classification datasets.
- **`generator.py`**: Iterates over the raw dataset for baseline generation.
- **`pipeline.py`**: Sequence processor for a single 3D CT classification sample.
- **`rotation.py`**: Applies standalone rotations for baseline data augmentation.
- **`tube.py`**: Injects synthetic ET tubes into baseline standalone volumes.

### `evaluation/`
- **`__init__.py`**: Python package initializer.
- **`evaluate.py`**: Calculates baseline classification metrics (Accuracy, F1, ROC-AUC) without spatial CC analysis.
- **`visuals.py`**: Generates binary prediction comparison charts for the baseline classifier.

### `lib/`
- **`__init__.py`**: Python package initializer.
- **`nifti_io.py`**: Mirrored helpers tailored for the baseline's distinct directory paths.

### `segmentation/`
- **`seg_generator.py`**: Generates organ segmentations specific for the baseline data generation.

### `training/` (Baseline Deep Learning)
- **`__init__.py`**: Python package initializer.
- **`data_loader.py`**: Fetches standalone classification images instead of sequential pairs.
- **`model_io.py`**: Manages baseline classification model weights.
- **`models.py`**: Contains the standard ResNet/VGG baseline architectures.
- **`trainer.py`**: Standard binary-cross-entropy (BCE) training loop designed for standard image classification.

---

## 3. 🎮 Demos (`demos/`)
This directory contains the code to execute clinical inference using pre-trained weights on both pre-generated synthetic datasets and real-world clinical datasets.

### `real_data_demo/` (Demo 3)
Evaluates the model on chronologically ordered, real-world longitudinal patient visits from the ICU.
- **`run_all.py`**: Orchestrator that triggers inference on the real-world COVID-19 dataset for both Tubes and Consolidations.
- **`consolidation_pkg/`**:
  - **`preprocessing.py`**: Specialized processing to align, resize, and normalize raw clinical DICOM/JPEG images for consolidations.
  - **`run_consolidation.py`**: Executes model inference specifically for real-world lung consolidations.
- **`lib/`**:
  - **`__init__.py`**: Python package initializer.
  - **`overlay_heatmap.py`**: Helper to overlay generated heatmaps onto the real X-rays for visual grids.
- **`tube_pkg/`**:
  - **`preprocessing.py`**: Specialized processing to align, resize, and normalize raw clinical DICOM/JPEG images for tubes.
  - **`run_tube.py`**: Executes model inference specifically for real-world endotracheal tubes.

### `synthetic_data_demo/` (Demo 2)
Evaluates the model on procedurally generated synthetic 2D DRR X-ray pairs.
- **`config.py`**: Defines input/output paths and limits for the synthetic test set.
- **`pull_data.py`**: A helper script to fetch synthetic test sets if they are not stored locally.
- **`run_all.py`**: Orchestrator that triggers inference on synthetic datasets for both Tubes and Consolidations.
- **`consolidation_pkg/`**:
  - **`preprocessing.py`**: Handles loading and formatting synthetic paired data arrays.
  - **`run_consolidation.py`**: Executes inference for lung consolidations on synthetic test splits.
- **`lib/`**:
  - **`__init__.py`**: Python package initializer.
  - **`overlay_heatmap.py`**: Helper to overlay generated heatmaps onto the synthetic X-rays.
- **`tube_pkg/`**:
  - **`preprocessing.py`**: Handles loading and formatting synthetic paired data arrays for ET tube tasks.
  - **`run_tube.py`**: Executes inference for endotracheal tubes on synthetic test splits.

---

## 4. 📂 Real Data Acquisition (`data/real_xray/`)
Tools for fetching and filtering external clinical datasets.

- **`pull_data.py`**: A standalone dataset parser. It connects to the public IEEE8023 COVID-19 Chest X-ray repository, downloads the clinical `metadata.csv`, and programmatically filters patients to find longitudinal sequences exhibiting dynamic intubations or consolidations. It then automatically downloads the corresponding raw X-ray images and clinical notes into structured directories for testing in Demo 3.
