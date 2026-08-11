# 📂 Codebase Structure Guide

This guide explains the architecture of the codebase in detail and the specific purpose of each module. The project is split into several highly specialized packages that handle end-to-end data generation, training, and inference.

## ⚙️ Global Configuration
- **`config.py`**: The universal configuration file. It contains all absolute paths, hyperparameters, tunable constants, and toggle flags used universally across every module.
- **`user_settings.py`**: A high-level, user-friendly wrapper around the configurations, allowing users to quickly toggle execution modes without diving into the codebase.

## 🧬 Data Generation (`datagen/`)
This package is responsible for procedural synthesis, inserting anomalies into 3D CT scans, and applying 3D-to-2D physics projections.

- **`generator.py`**: The top-level orchestrator for data generation. It handles iterating over the raw datasets, managing files, and passing data to the pipeline.
- **`pipeline.py`**: Executes the complete sequential data generation logic for a single CT scan pair. It carefully manages memory to ensure large 3D volumes do not overflow the GPU.
- **`tube.py` & `tube_randomization.py`**: Handles the mathematical insertion, positioning, and randomization of synthetic Endotracheal tubes within the tracheal segmentation boundaries using Euclidean Distance Transforms (EDT) and directed random walks.
- **`deformation.py`, `pooling.py`, & `shapes.py`**: Responsible for the generation of lung consolidations. They create base spheres, apply B-spline deformations for organic irregularity, and use 3D pooling to smooth the surfaces realistically.
- **`crop.py`**: Optimizes processing by isolating the relevant lung regions via tight bounding-box cropping on the 3D CT.
- **`rotation.py`**: Applies complex 3D volume rotations independently to prior and current scans to simulate inter-scan patient positioning and posture shifts.
- **`noise.py`**: Injects statistical noise into the synthetic output to simulate real-world X-ray sensor artifacts.
- **`drr.py`**: Applies Digitally Reconstructed Radiography (DRR). It mathematically collapses 3D CT volumes into 2D synthetic X-ray projections via depth-axis ray-summing.

## 🧠 Neural Network Training (`training/`)
This package manages the deep learning architecture and the optimization loop for the Twin-CNN model.

- **`models.py`**: Defines the `VGGDiffNet` architecture, including the weight-sharing twin encoders and the subtraction bottleneck designed to learn rotation-invariant feature representations.
- **`trainer.py`**: Contains the core PyTorch training loop, Mean Squared Error (MSE) loss logic, and mixed-precision gradient scaling for efficient GPU training.
- **`data_loader.py`**: Implements custom PyTorch Dataset classes and Dataloaders for efficient, batched loading of the massively generated 2D X-ray image pairs.
- **`model_io.py`**: Manages the saving, loading, and checkpointing of trained neural network weights.

## 📈 Evaluation (`evaluation/`)
This package handles the quantitative and qualitative assessment of the models on both synthetic and real-world data.

- **`evaluate.py`**: The main evaluation orchestrator that runs the trained models against the unseen test splits.
- **`cc_lib.py` & `cc_metrics.py`**: Implements the Object-Level Connected Components (CC) evaluation. It handles image thresholding, blob extraction, spatial Intersection-over-Union (IoU) mapping, and calculating True Positives/False Negatives to prevent unfair penalization of minor 2D projection shifts.
- **`visuals.py`**: Generates intuitive side-by-side PNG grids comparing the prior X-ray, current X-ray, ground truth heatmap, and the model's predicted heatmap.
- **`pw_metrics.py`**: Handles standard strict Pixel-Wise (PW) mathematical metrics as a baseline comparison.

## 🚀 Execution Scripts
- **`run_datagen.py`**: Standalone script to trigger only the 3D-to-2D data generation pipeline.
- **`run_training.py`**: Standalone script to trigger only the PyTorch network training loop.
- **`run_inference.py`**: The clinical evaluation script that loads `.pth` weights and predicts changes on either synthetic pairs or chronological real-world patient data.
- **`run_all.py`**: The master orchestrator that chains all the above scripts together based on `user_settings.py`.
