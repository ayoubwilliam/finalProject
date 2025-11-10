# Detecting Air Under the Diaphragm in Chest X-Ray Images

**Engineering Project - Group 104: William Ayoub and Agam Hershko**  
**Hebrew University of Jerusalem (HUJI) - CASMIP Lab**

---

## 📌 Overview

This project aims to develop a deep learning system to automatically detect pneumoperitoneum (air under the diaphragm) in chest X-ray images, specifically by comparing consecutive X-rays of the same patient in Intensive Care Unit (ICU) settings.

Detecting small volumes of free air is a critical, time-sensitive task often hampered by variable image quality, patient positioning, and overlapping anatomy in portable ICU radiographs. Our solution leverages a Twin Convolutional Neural Network (CNN) architecture trained primarily on synthetic data generated from 3D CT scans to identify these subtle temporal changes.

---

## 🚀 Key Features

- **Temporal Comparison**: Compares prior and posterior X-rays to highlight new findings.
- **Synthetic Data Generation**: Utilizes 3D CT scans to generate realistic 2D Digitally Reconstructed Radiographs (DRRs), allowing for controlled simulation of pathologies (like free air).
- **Twin CNN Architecture**: A specialized deep learning model designed to process paired images and output a "difference heatmap."
- **Automated Segmentation**: Integrates tools like TotalSegmentator for precise anatomical awareness during synthetic data creation.

---

## 🛠️ Architecture & Methodology

Our approach tackles the scarcity of labeled paired X-ray data by creating a synthetic training pipeline:

### Synthetic Data Unit:
1. **Input**: 3D CT Scans.
2. **Manipulation**: Artificially inserting "air" or other changes into the 3D volume.
3. **Projection**: Generating 2D DRRs (simulated X-rays) from both original and manipulated CTs, controlling for patient pose and angle.

### Detection Unit (Backend):
- A **Twin CNN** receives the paired images (prior/posterior).
- The network learns spatial differences and outputs a heatmap localizing the suspected pneumoperitoneum.

---

## 📂 Repository Structure

```
├── drr_output/          # Directory for generated synthetic X-rays (DRRs)
├── segmentations/       # Outputs from anatomical segmentation (TotalSegmentator)
├── drr.py               # Core script for generating Digitally Reconstructed Radiographs
├── file_handler.py      # Utilities for loading CT scans (NiBabel wrappers) and managing I/O
├── rotation.py          # Handling 3D rotations and pose adjustments for DRR generation
├── segmentation.py      # Interface for running 3D segmentation tasks
├── LICENSE
└── README.md
```

---

## 🔧 Technologies

- **Deep Learning**: PyTorch  
- **Medical Imaging Processing**: NiBabel, TotalSegmentator  
- **Numerical Operations**: NumPy  
- **Visualization**: Matplotlib  

---

## 💻 Getting Started

### Prerequisites

- **Python**: 3.8+  
- **CUDA-capable GPU** (recommended for DRR generation and training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies (consider using a virtual environment):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
   pip install nibabel totalsegmentator matplotlib numpy
   ```

---

### Basic Usage (Synthetic Data Generation)

**(Update this section with actual usage examples as you develop the CLI)**

To generate a DRR from a CT scan:
```bash
python drr.py --input data/sample_ct.nii.gz --output drr_output/sample_xray.png
```

---

## 👥 Team

### Students:
- **Agam Hershko** ([agam.hershkodek@mail.huji.ac.il](mailto:agam.hershkodek@mail.huji.ac.il))  
- **William Ayoub** ([william.ayoub@mail.huji.ac.il](mailto:william.ayoub@mail.huji.ac.il))

### Advisors (CASMIP Lab):
- **Prof. Leo Joskowicz**  
- **Itamar Sabban**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
