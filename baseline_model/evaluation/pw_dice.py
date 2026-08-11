"""
Pixel-wise Dice evaluation.
Calculates independent Red (positive) and Green (negative) Dice scores across the test set.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from training.model_io import load_active_model, predict_heatmap
from training.data_loader import get_test_dataloader


def calculate_dice_score(preds_flat, targets_flat):
    """Calculates Red, Green, and Mean Dice scores for a flattened image pair."""
    p_red = (preds_flat > cfg.PW_DICE_THRESHOLD).astype(int)
    t_red = (targets_flat > cfg.PW_DICE_THRESHOLD).astype(int)

    p_green = (preds_flat < -cfg.PW_DICE_THRESHOLD).astype(int)
    t_green = (targets_flat < -cfg.PW_DICE_THRESHOLD).astype(int)

    def _dice(pm, tm):
        denom = np.sum(pm) + np.sum(tm)
        return 1.0 if denom == 0 else (2.0 * np.sum(pm * tm)) / denom

    d_red = _dice(p_red, t_red)
    d_green = _dice(p_green, t_green)

    return {
        "Dice_Red": d_red,
        "Dice_Green": d_green,
        "Dice_Mean": (d_red + d_green) / 2.0,
    }


def run_pw_dice():
    """Iterates the test dataset to generate the Pixel-Wise Dice report."""
    print(f"\n[INFO] Starting Pixel-Wise Dice Evaluation (Threshold={cfg.PW_DICE_THRESHOLD})...")
    model = load_active_model()
    test_loader, _ = get_test_dataloader()
    aggregator = {}

    with torch.no_grad():
        for prior, current, target_angle, filenames, target in tqdm(test_loader, desc="Calculating PW-Dice"):
            preds_np = predict_heatmap(model, prior.to(cfg.DEVICE), current.to(cfg.DEVICE)).cpu().numpy()
            target_np = target.numpy()

            for i in range(preds_np.shape[0]):
                single_dice = calculate_dice_score(preds_np[i].flatten(), target_np[i].flatten())
                for k, v in single_dice.items():
                    aggregator.setdefault(k, []).append(v)

    os.makedirs(os.path.dirname(cfg.EVAL_FILE_PW_DICE), exist_ok=True)
    with open(cfg.EVAL_FILE_PW_DICE, "w") as f:
        f.write(f"PIXEL-WISE DICE REPORT | Model: {cfg.SELECTED_MODEL}\n")
        f.write(f"Threshold: {cfg.PW_DICE_THRESHOLD}\n")
        f.write("-" * 30 + "\n")
        for k, v in {k: np.mean(v) for k, v in aggregator.items()}.items():
            f.write(f"{k:<15}: {v:.4f}\n")

    print(f"[INFO] Metrics saved to {cfg.EVAL_FILE_PW_DICE}")
