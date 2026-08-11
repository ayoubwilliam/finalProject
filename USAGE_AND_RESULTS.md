# 🚀 Usage, Evaluation & Results Guide

This document explains exactly what is included in the submitted package, the three main demos you can run, what the one-liner command does behind the scenes, and exactly where to find the generated results with expected file trees.

---

## 🧪 Provided Demos in the Package

> [!WARNING]
> **Crucial Note Regarding Data Volume & Model Performance**
> While this repository contains the **full, original End-to-End codebase**, the provided `Runnable_Version` only includes a heavily limited subset of raw CT scans (5 files). 
>
> The *highly optimized* pretrained models included in this package were trained on a massive dataset of over **2,500 CT files (~300GB of data)**. Hosting a 300GB dataset on GitHub is physically impossible, and distributing it requires strict ethical permissions from the CASMIP lab. 
> 
> Therefore, while **Demo 1** successfully demonstrates the complete training pipeline mechanics, the model trained dynamically during this demo will inevitably suffer in performance compared to the pretrained weights due to the severe lack of training data. You can partially alleviate this by increasing the `NUMBER_OF_PAIRS` parameter in `user_settings.py` to artificially expand the dataset via augmentation, but this is not a complete solution for replacing the full 300GB dataset.

The `Runnable_Version` you download contains a carefully curated subset of data (to keep the download size manageable) designed to demonstrate the complete capabilities and evaluation metrics of our system. 

By executing the master orchestrator (`run_all.py`), you are actively running **3 main demos**:

### Demo 1: End-to-End Synthetic Pipeline (Data Gen & Training)
This demo operates on a limited set of raw 3D CT files to procedurally generate synthetic 2D DRR X-ray pairs (both healthy and anomalous), dynamically trains the PyTorch VGGDiffNet model from scratch on your GPU, and evaluates the results.
- **Input Data:** `data/ct/`
- **Output Path:** `output/synthetic_pairs/` and `output/checkpoints/`

### Demo 2: Clinical Inference on Pre-Generated Synthetic Data
This demo bypasses the heavy training step. It loads our highly-optimized **Pretrained Weights** and runs rapid clinical inference on a pre-generated subset of synthetic anomalous X-rays, strictly evaluating the model's performance.
- **Input Data:** `data/synthetic_xray/synthetic xray/`
- **Output Path:** `output/pretrained_model_results/synthetic_xray/`

### Demo 3: Clinical Inference on Real-World Patient Data
This demo applies our Pretrained Models to **real, chronological patient X-ray visits** from the ICU. It compares their sequential scans, overlays the model's predicted anomaly heatmaps, and prints out the actual clinical physician notes alongside the visual results.
- **Input Data:** `data/real_xray/consolidation_tube_pairs/`
- **Output Path:** `output/pretrained_model_results/real_xray/`

---

## 📦 Expected Output & File Trees

When you execute the master orchestrator, the codebase creates a highly structured `output/` directory locally to safely isolate all generated files, AI models, and visual results. 

Here is exactly what the file tree will look like after running the demos, and where you can find everything:

```text
Runnable_Version/
│
├── data/                       # Contains the limited subset of input data for the demos
│
└── output/                     # 🚀 ALL RESULTS ARE SAVED HERE!
    │
    ├── synthetic_pairs/        # (Demo 1) Generated X-Rays
    │   ├── train/
    │   └── test/
    │       └── scan_001/
    │           ├── prior.nii.gz         # Baseline simulated X-ray
    │           ├── current.nii.gz       # Simulated X-ray with injected anomaly
    │           └── heatmap_gt.nii.gz    # Ground Truth mask
    │
    ├── checkpoints/            # (Demo 1) Trained Model Weights
    │   ├── best_model.pth      # Lowest validation loss weights
    │   └── last_model.pth
    │
    ├── results/                # (Demo 1) Training Pipeline Evaluation
    │   ├── metrics.csv         # Strict F1-Score, CC, and Dice scores
    │   └── visuals/            # Visual grids of True vs Predicted heatmaps
    │
    └── pretrained_model_results/ # (Demo 2 & 3) Pretrained Inference Results
        ├── synthetic_xray/
        │   ├── consolidation/
        │   │   └── scan_XXX_consolidation_output.png # Heatmap comparison grid
        │   └── tube/
        │       └── scan_XXX_tube_output.png          # Heatmap comparison grid
        │
        └── real_xray/
            ├── consolidation/
            │   ├── patient_XXX_consolidation_output.png
            │   └── patient_XXX_consolidation_output.txt  # Clinical notes & status
            └── tube/
                ├── patient_XXX_tube_output.png
                └── patient_XXX_tube_output.txt           # Clinical notes & status
```

---

## ⚡ The One-Liner Execution

The quickest way to run all 3 demos simultaneously is using the terminal one-liner provided in the `README.md`. 
Here is exactly what happens when you paste that command into your terminal:

1. **`curl` / Download**: It automatically reaches out to GitHub Releases and downloads the `Runnable_Version.zip` package locally.
2. **`zipfile` / Extraction**: It uses Python's native zip engine to perfectly extract the folder across Windows, macOS, or Linux.
3. **`venv` / Sandboxing**: It creates a completely isolated "Virtual Environment" (`venv`). This is crucial because it ensures that installing heavy deep-learning dependencies won't corrupt your system-wide Python installation.
4. **`pip` / Dependencies**: It installs `torch`, `numpy`, `TotalSegmentator`, and other dependencies strictly inside the sandbox.
5. **`cupy` / Dynamic GPU Acceleration**: It executes a smart Python script that interrogates your PyTorch installation to find your specific CUDA version, and dynamically installs the exactly matching `cupy` package (e.g. `cupy-cuda12x`) for maximum GPU acceleration.
6. **Execution**: Finally, it triggers the `run_all.py` master orchestrator inside the sandbox!

---

## 🎮 How to Run Manually

If you prefer to run the codebase manually (or want to toggle specific demos on or off), follow these steps:

1. **Configure Settings**: Open `user_settings.py` in any text editor. You can easily toggle `True/False` flags to choose whether to run the Data Generation & Training pipeline (`RUN_PIPELINES = True`) or the Pretrained Clinical Inference (`RUN_PRETRAINED = True`).
2. **Execute Orchestrator**: Run the main script via terminal:
   ```bash
   python run_all.py
   ```
   *(Ensure you are running this inside your virtual environment so the dependencies are recognized!)*
