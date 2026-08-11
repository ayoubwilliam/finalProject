# Finding Changes of Interest in Chest X-rays in ICU
**Hebrew University of Jerusalem - Final Project**  
**Group 104**: Agam Hershko & William Ayoub  
**Advisor:** Prof. Leo Joskowicz (CASMIP Lab) | **Mentor:** Gal Fiebelman

## 🚀 Quick Start (One-Liner)

We have provided a fully packaged Runnable Version of this project via GitHub Releases. You can download the packaged zip, extract it, dynamically install all dependencies, and run the complete End-to-End Orchestrator (Inference & Training) using a single command in your terminal.

Assuming you have **Python 3.11** installed, simply copy and paste the command that matches your operating system:

**For Windows (`python` & `pip`):**
```bash
curl -L -o Runnable_Version.zip https://github.com/ayoubwilliam/finalProject/releases/download/v1.0.0/Runnable_Version.zip && python -c "import zipfile; zipfile.ZipFile('Runnable_Version.zip', 'r').extractall('.')" && cd Runnable_Version && python -m venv venv && venv\Scripts\pip install -r requirements.txt && venv\Scripts\python -c "import torch, os; v=torch.version.cuda; os.system(r'venv\Scripts\pip install cupy-cuda' + v.split('.')[0] + 'x' if v else r'venv\Scripts\pip install cupy')" && venv\Scripts\python run_all.py
```

**For Linux / macOS (`python3` & `pip3`):**
```bash
curl -L -o Runnable_Version.zip https://github.com/ayoubwilliam/finalProject/releases/download/v1.0.0/Runnable_Version.zip && python3 -c "import zipfile; zipfile.ZipFile('Runnable_Version.zip', 'r').extractall('.')" && cd Runnable_Version && python3 -m venv venv && venv/bin/pip install -r requirements.txt && venv/bin/python -c "import torch, os; v=torch.version.cuda; os.system('venv/bin/pip install cupy-cuda' + v.split('.')[0] + 'x' if v else 'venv/bin/pip install cupy')" && venv/bin/python run_all.py
```

### What this command does:
1. Downloads the `Runnable_Version.zip` from the GitHub Releases page using `curl`.
2. Extracts the zip file completely securely using Python's built-in `zipfile` module.
3. Installs the bulk of dependencies (`torch`, `numpy`, `TotalSegmentator`, etc.) from `requirements.txt`.
4. Automatically detects your PyTorch CUDA version and dynamically installs the correctly matching `cupy` GPU acceleration package. *(Note: While this handles other CUDA versions, the pipeline was extensively tested on **CUDA 12**, and using CUDA 12 is strongly recommended for optimal stability!)*
5. Executes the master `run_all.py` orchestrator.

---

## 📂 Code Structure & Documentation
Want to dive into the codebase? We have thoroughly documented the purpose, architecture, and logic of every single script in the project!

👉 **[Click here to view the detailed Codebase Structure Guide](CODEBASE_GUIDE.md)**

---

## 🩻 Abstract
In the Intensive Care Unit (ICU), mechanically ventilated patients require continuous monitoring. Detecting critical temporal changes between consecutive chest X-rays, such as endotracheal tube shifts or new lung consolidations, is essential but highly error-prone. 

To overcome the severe scarcity of paired medical X-ray data, we engineered a fully automated **twin Convolutional Neural Network (CNN) pipeline trained entirely on synthetic data**. We procedurally insert anomalies (tubes and consolidations) into healthy 3D CT scans, apply 3D rotations to simulate realistic patient positioning, and use Digitally Reconstructed Radiography (DRR) to project them into synthetic 2D chest X-rays.

A weight-sharing twin-encoder CNN processes both X-rays into a rotation-invariant latent space, using a subtraction bottleneck to perfectly align healthy anatomy and isolate true pathological differences.

---

## ✨ Core Features
- **Procedural Synthetic Data Generation:** Generates thousands of 2D X-ray pairs (Prior & Current) from base 3D CT scans using physical DRR ray-summing projections.
- **Twin-Encoder CNN Architecture:** A VGG-style subtraction bottleneck architecture that natively ignores 3D patient rotation and posture shifts.
- **Automated Anomaly Injection:** Clinically realistic procedural generation of Endotracheal Tubes (using random walks and distance transforms) and Lung Consolidations (using B-spline deformations).
- **Connected Components Evaluation:** Object-level spatial accuracy evaluation ignoring minor 2D projection discrepancies.



---

## ⚙️ Configuration
The master orchestrator reads directly from `user_settings.py`. Open `user_settings.py` to easily toggle boolean flags for:
- Data Generation targets (Consolidations vs Tubes)
- Number of synthetic pairs to generate per CT scan
- Whether to execute the heavy data generation/training pipeline
- Whether to run inference using our provided Pretrained Models

---

## 📊 Architecture and Results
- **Data:** Trained on over 12,500 unique synthetic pairs procedurally generated from public 3D CT datasets.
- **Evaluation:** Achieved F1 classification scores of **86.56%** (Consolidations) and **85.85%** (Endotracheal Tubes).
- **Spatial Accuracy:** Object-level connected components evaluation yielded an outstanding **>90% average Dice score** on spatial detection.
- **Clinical Validation:** Despite being trained *entirely* on synthetic data, the models successfully identified worsening clinical conditions and device migrations in real-world COVID-19 patient sequences.
