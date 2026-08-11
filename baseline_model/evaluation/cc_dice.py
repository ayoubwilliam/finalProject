"""
Connected components instance-level Dice evaluation.
Collects per-object Dice scores for TP matches across the test set.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from training.model_io import load_active_model, predict_heatmap
from training.data_loader import get_test_dataloader
from evaluation import cc_lib


def _accumulate_dice_scores(model, test_loader):
    """Runs inference to collect raw Instance Dice scores."""
    r_tp_only = []
    g_tp_only = []

    with torch.no_grad():
        for prior_b, current_b, target_angle, filenames, target_b in tqdm(test_loader, desc="Calculating Instance Dice"):
            pred_b = predict_heatmap(model, prior_b.to(cfg.DEVICE), current_b.to(cfg.DEVICE))

            for i in range(prior_b.shape[0]):
                analysis = cc_lib.analyze_single_sample(
                    pred_b[i].squeeze(),
                    target_b[i].squeeze().to(cfg.DEVICE),
                    cfg.PRED_THRESHOLD, cfg.GT_THRESHOLD, cfg.IOU_THRESHOLD,
                )
                r_tp_only.extend(analysis.r_dice_tp_only)
                g_tp_only.extend(analysis.g_dice_tp_only)

    return r_tp_only, g_tp_only


def _safe_mean(score_list):
    """Returns the mean of a list, or 0.0 if empty."""
    return np.mean(score_list) if score_list else 0.0


def _format_dice_report(means, num_samples):
    """Formats the mean scores into a readable text report."""
    f_r_tp, f_g_tp = means

    return f"""==================================================
INSTANCE-LEVEL DICE EVALUATION
Model: {cfg.SELECTED_MODEL}
Test Samples: {num_samples}
==================================================

[RED / POSITIVE CHANGES]
Mean Dice (TP Matches Only)   -> {f_r_tp:.4f}

--------------------------------------------------
[GREEN / NEGATIVE CHANGES]
Mean Dice (TP Matches Only)   -> {f_g_tp:.4f}

==================================================
[OVERALL COMBINED PERFORMANCE]
Overall Dice (TP Matches Only)-> {(f_r_tp + f_g_tp) / 2.0:.4f}
=================================================="""


def run_cc_dice():
    """Main pipeline for Instance-Level Dice scores."""
    print(f"\n[INFO] Running Instance-Level Dice Metrics on {cfg.DEVICE}...")

    model = load_active_model()
    test_loader, test_files = get_test_dataloader()

    r_tp, g_tp = _accumulate_dice_scores(model, test_loader)

    means = (_safe_mean(r_tp), _safe_mean(g_tp))
    report = _format_dice_report(means, len(test_files))

    os.makedirs(os.path.dirname(cfg.EVAL_FILE_CC_DICE), exist_ok=True)
    with open(cfg.EVAL_FILE_CC_DICE, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"[SUCCESS] Saved to: {cfg.EVAL_FILE_CC_DICE}")
