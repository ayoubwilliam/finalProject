"""
Pixel-wise ROC/AUC evaluation.
Collects all predictions, computes absolute magnitude ROC curve, and saves the plot.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import config as cfg
from training.model_io import load_active_model, predict_heatmap
from training.data_loader import get_test_dataloader


def run_pw_auc():
    """Calculates and plots the Unified Absolute ROC curve for the test set."""
    print(f"\n[INFO] Collecting predictions for Unified ROC Curve on {cfg.DEVICE}...")
    model = load_active_model()
    test_loader, _ = get_test_dataloader()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for prior, current, target_angle, filenames, target in tqdm(test_loader, desc="Inference for ROC"):
            preds = predict_heatmap(model, prior.to(cfg.DEVICE), current.to(cfg.DEVICE))
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(target.numpy().flatten())

    y_scores_abs = np.abs(np.concatenate(all_preds))
    t_binary = (np.abs(np.concatenate(all_targets)) > cfg.MATH_EPSILON).astype(int)

    fpr, tpr, thresholds = roc_curve(t_binary, y_scores_abs)
    roc_auc = auc(fpr, tpr)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    plt.figure(figsize=cfg.PLOT_FIGSIZE_ROC)
    plt.plot(fpr, tpr, color='purple', lw=2, label=f'Unified Magnitude (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Unified ROC Curve (Absolute Magnitude)\nModel: {cfg.SELECTED_MODEL}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    os.makedirs(cfg.EVAL_PW_DIR, exist_ok=True)
    plt.savefig(cfg.EVAL_FILE_PW_ROC, dpi=cfg.PLOT_DPI_HIGH)
    plt.close()

    print(f"[SUCCESS] ROC Curve saved to {cfg.EVAL_FILE_PW_ROC}")
    print(f"          OPTIMAL THRESHOLD: {best_thresh:.6f}")
