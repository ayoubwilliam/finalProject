# 🚀 Finding Changes of Interest in Chest X-rays in ICU
**Hebrew University of Jerusalem - Final Project**  
**Group 104**: Agam Hershko (`agam.hershkodek@mail.huji.ac.il`) & William Ayoub (`william.ayoub@mail.huji.ac.il`)  
**Advisor:** Prof. Leo Joskowicz (CASMIP Lab) | **Mentor:** Gal Fiebelman

---

## 🩻 Abstract
Detecting critical temporal changes between consecutive chest X-rays in the ICU (like endotracheal tube shifts or new lung consolidations) is essential but highly error-prone. To overcome severe medical data scarcity, we engineered an automated **twin Convolutional Neural Network (CNN) pipeline trained entirely on synthetic data**. We procedurally inject anomalies into healthy 3D CT scans, apply 3D posture rotations, and project them into synthetic 2D chest X-rays using Digitally Reconstructed Radiography (DRR). A weight-sharing twin-encoder CNN processes these X-rays to perfectly align healthy anatomy and isolate true pathological differences.

---

## 📚 Codebase Architecture
For a detailed breakdown of the internal file structure and the role of every single python script, please see the [**Codebase Structure Guide**](CODEBASE_GUIDE.md).

---

## 🚀 Running the Demos - One liner quick start
*(Runs all 3 Demos included in this package - See below!)*

Assuming you have Python 3.11 installed, simply copy and paste the command that matches your operating system:

**For Windows (`cmd`):**
```bash
curl.exe -L -o Runnable_Version.zip https://github.com/ayoubwilliam/finalProject/releases/download/v1.0.0/Runnable_Version.zip && py -3.11 -c "import zipfile; zipfile.ZipFile('Runnable_Version.zip', 'r').extractall('.')" && cd Runnable_Version && py -3.11 -m venv venv && .\venv\Scripts\pip install -r requirements.txt && .\venv\Scripts\python -c "import torch, os, sys; v=torch.version.cuda; cmd=chr(34)+sys.executable+chr(34)+' -m pip install cupy'; os.system(cmd+'-cuda'+v.split('.')[0]+'x' if v else cmd)" && .\venv\Scripts\python run_all.py
```

**For Linux / macOS (`python3` & `pip3`):**
```bash
curl -L -o Runnable_Version.zip https://github.com/ayoubwilliam/finalProject/releases/download/v1.0.0/Runnable_Version.zip && python3 -c "import zipfile; zipfile.ZipFile('Runnable_Version.zip', 'r').extractall('.')" && cd Runnable_Version && python3 -m venv venv && venv/bin/pip install -r requirements.txt && venv/bin/python -c "import torch, os, sys; v=torch.version.cuda; cmd=chr(34)+sys.executable+chr(34)+' -m pip install cupy'; os.system(cmd+'-cuda'+v.split('.')[0]+'x' if v else cmd)" && venv/bin/python run_all.py
```

We have provided a fully packaged Runnable Version of this project via GitHub Releases. You can download the packaged zip, extract it, dynamically install all dependencies, and run the complete End-to-End Orchestrator (Inference & Training) using a single command in your terminal.

> [!IMPORTANT]
> **Python 3.11 is STRICTLY REQUIRED.** 
> Due to strict native C++/CUDA bindings and pinned dependencies, other versions (like 3.10 or 3.12) will fail to build!
> *(Note: While the one-liner dynamically handles other CUDA versions, the pipeline was extensively tested on **CUDA 12**, and using CUDA 12 is strongly recommended for optimal stability!)*

> [!WARNING]
> **Data Volume & Model Performance:** The provided `Runnable_Version` includes a limited subset of 5 raw CT scans (to fit on GitHub). While **Demo 1** fully demonstrates the training pipeline, its resulting model will perform worse than our included **Pretrained Models**, which were optimized on **2,500 CT files (~300GB)**. You can increase `NUMBER_OF_PAIRS` in `user_settings.py` for more augmented data, but the pretrained weights remain far superior.

> [!NOTE]
> **Execution Time:** Please allow ~**30 minutes (on an RTX 4090)** for the one-liner to download heavy dependencies, train, and run inference across all 3 demos.

*(What this one-liner does: Downloads the package, securely installs all dependencies in an isolated virtual environment, and runs all 3 demos!)*

---

## 🧪 Provided Demos in the Package

The `Runnable_Version` you download contains a carefully curated subset of data (to keep the download size manageable) designed to demonstrate the complete capabilities and evaluation metrics of our system. 

By executing the master orchestrator (`run_all.py`), you are actively running **3 main demos**:

### Demo 1: End-to-End Synthetic Pipeline (Data Gen & Training)
This demo operates on a limited set of raw 3D CT files to procedurally generate synthetic 2D DRR X-ray pairs (both containing injected anomalies to simulate temporal changes), dynamically trains the PyTorch VGGDiffNet model from scratch on your GPU, and evaluates the results.
- **Input Data:** `data/ct/`
- **Output Path:** `output/synthetic_pairs/` and `output/checkpoints/`

### Demo 2: Clinical Inference on Pre-Generated Synthetic Data
This demo bypasses the heavy training step. It loads our highly-optimized **Pretrained Weights**—which were painstakingly trained on the full 2,500 CT (300GB) dataset, yielding vastly higher quality and accuracy than the model produced in Demo 1. It runs rapid clinical inference on a pre-generated subset of synthetic anomalous X-rays, strictly evaluating the model's performance.
- **Input Data:** `data/synthetic_xray/synthetic xray/`
- **Output Path:** `output/pretrained_model_results/synthetic_xray/`

### Demo 3: Clinical Inference on Real-World Patient Data
This demo applies the same high-quality, 300GB-trained Pretrained Models to **real, chronological patient X-ray visits** from the ICU. This curated demo includes **27 patients from our COVID-19 dataset** (as detailed in the final project report). It compares their sequential scans, overlays the model's predicted anomaly heatmaps, and prints out the actual clinical physician notes alongside the visual results. *(Note: The exemplary patient featured in our final PDF report is **Patient 299**!)*
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
│   └── real_xray/              
│       └── pull_data.py        # ⬇️ Script to automatically download the real clinical X-ray dataset!
│
└── output/                     # 🚀 ALL RESULTS ARE SAVED HERE!
    │
    ├── generated_synthetic/    # (Demo 1) Generated X-Rays (Grouped by original CT)
    │   ├── ct_file1/           
    │   │   ├── Pair1/          # Each pair contains: prior, current, and heatmap_gt
    │   │   ├── Pair2/
    │   │   └── Pair3/
    │   └── ct_file2/
    │
    ├── model_weights/          # (Demo 1) Trained Model Weights
    │   └── trained_model_weights.pth   # The final trained model weights
    │
    ├── evaluation/             # (Demo 1) Training Pipeline Evaluation
    │   ├── CC/                 # Connected Components Analysis
    │   │   └── Dataset_Metrics/# Object-level Dice & evaluation metrics
    │   └── Visuals/            # Visual grids (True vs Predicted heatmaps) and .nii.gz outputs
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

## 🎮 How to Run Manually

If you prefer to run the codebase manually (or want to toggle specific demos on or off), follow these steps:

1. **Configure Settings**: Open `user_settings.py` in any text editor. You can easily toggle `True/False` flags to choose whether to run the Data Generation & Training pipeline (`RUN_PIPELINES = True`) or the Pretrained Clinical Inference (`RUN_PRETRAINED = True`).
2. **Execute Orchestrator**: Run the main script via terminal:
   ```bash
   python run_all.py
   ```
   *(Ensure you are running this inside your virtual environment so the dependencies are recognized!)*

---

## 📊 Architecture and Results
- **Data:** Trained on over 12,500 unique synthetic pairs procedurally generated from public 3D CT datasets.
- **Evaluation:** Achieved F1 classification scores of **86.56%** (Consolidations) and **85.85%** (Endotracheal Tubes).
- **Spatial Accuracy:** Object-level connected components evaluation yielded an outstanding **>90% average Dice score** on spatial detection.
- **Clinical Validation:** Despite being trained *entirely* on synthetic data, the models successfully identified worsening clinical conditions and device migrations in real-world COVID-19 patient sequences.
