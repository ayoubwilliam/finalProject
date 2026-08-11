"""
Standard evaluation visualizations.
Generates 4-panel overlays (Prior, Current, GT, Prediction) and saves predictions as NIfTI.
"""

import os
import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg
from training.model_io import load_active_model, predict_heatmap
from training.data_loader import get_test_dataloader
from lib.overlay_heatmap import build_overlay_colormap, plot_heatmap_overlay, plot_base_image


def run_standard_visuals():
    """Generates heatmap overlay panels and saves raw prediction NIfTI files."""
    print(f"\n[INFO] Starting Standard Visual Evaluation...")
    os.makedirs(cfg.EVAL_VISUALS_DIR, exist_ok=True)

    model = load_active_model()
    test_loader, _ = get_test_dataloader()
    cmap = build_overlay_colormap()
    global_idx = 0

    with torch.no_grad():
        for prior, current, target_angle, filenames, target in tqdm(test_loader, desc="Saving Images & NIfTI"):
            prior = prior.to(cfg.DEVICE)
            current = current.to(cfg.DEVICE)

            preds_np = predict_heatmap(model, prior, current).cpu().numpy()
            prior_np = prior.cpu().numpy()
            current_np = current.cpu().numpy()
            target_np = target.numpy()

            for i in range(prior_np.shape[0]):
                name = filenames[i].replace(os.path.sep, "_")

                # 4-panel image
                fig, axes = plt.subplots(1, 4, figsize=cfg.PLOT_FIGSIZE_STANDARD)
                plot_base_image(axes[0], prior_np[i][0], "Prior", title_fontsize=12)
                plot_base_image(axes[1], current_np[i][0], "Current", title_fontsize=12)
                plot_heatmap_overlay(axes[2], current_np[i][0], target_np[i][0], "Ground Truth", cmap, fig,
                                     title_fontsize=12)
                plot_heatmap_overlay(axes[3], current_np[i][0], preds_np[i][0], "Prediction", cmap, fig,
                                     title_fontsize=12)

                plt.tight_layout()
                plt.savefig(os.path.join(cfg.EVAL_VISUALS_DIR, f"res_{global_idx:04d}_{name}.png"),
                            dpi=cfg.PLOT_DPI_STANDARD)
                plt.close(fig)

                # save prediction as NIfTI
                nifti_img = nib.Nifti1Image(preds_np[i][0], cfg.NIFTI_AFFINE)
                nib.save(nifti_img, os.path.join(cfg.EVAL_VISUALS_DIR, f"pred_{global_idx:04d}_{name}.nii.gz"))

                global_idx += 1
