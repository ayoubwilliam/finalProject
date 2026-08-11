"""
Connected components object-level metrics.
Accumulates TP/FP/FN counts across the test set and computes Precision/Recall/F1.
"""

import os
import torch
from tqdm import tqdm

import config as cfg
from training.model_io import load_active_model, predict_heatmap
from training.data_loader import get_test_dataloader
from evaluation import cc_lib


def _calculate_prf1(tp, fp, fn):
    """Safely calculates Precision, Recall, and F1-Score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _accumulate_object_counts(model, test_loader):
    """Runs inference to accumulate raw TP, FP, and FN counts."""
    r_tp, r_fp, r_fn = 0, 0, 0
    g_tp, g_fp, g_fn = 0, 0, 0

    with torch.no_grad():
        for prior_b, current_b, target_angle, filenames, target_b in tqdm(test_loader, desc="Calculating Object Counts"):
            pred_b = predict_heatmap(model, prior_b.to(cfg.DEVICE), current_b.to(cfg.DEVICE))

            for i in range(prior_b.shape[0]):
                analysis = cc_lib.analyze_single_sample(
                    pred_b[i].squeeze(),
                    target_b[i].squeeze().to(cfg.DEVICE),
                    cfg.PRED_THRESHOLD, cfg.GT_THRESHOLD, cfg.IOU_THRESHOLD,
                )
                r_tp += analysis.r_tp
                r_fp += analysis.r_fp
                r_fn += analysis.r_fn
                g_tp += analysis.g_tp
                g_fp += analysis.g_fp
                g_fn += analysis.g_fn

    return (r_tp, r_fp, r_fn), (g_tp, g_fp, g_fn)


def _format_metrics_report(r_counts, g_counts, num_samples):
    """Generates the comprehensive text report."""
    r_tp, r_fp, r_fn = r_counts
    g_tp, g_fp, g_fn = g_counts

    o_tp = r_tp + g_tp
    o_fp = r_fp + g_fp
    o_fn = r_fn + g_fn

    r_prec, r_rec, r_f1 = _calculate_prf1(r_tp, r_fp, r_fn)
    g_prec, g_rec, g_f1 = _calculate_prf1(g_tp, g_fp, g_fn)
    o_prec, o_rec, o_f1 = _calculate_prf1(o_tp, o_fp, o_fn)

    return f"""==================================================
OBJECT-LEVEL EVALUATION REPORT
Model: {cfg.SELECTED_MODEL}
Test Samples: {num_samples}
IoU Threshold: {cfg.IOU_THRESHOLD}
Pred Threshold: {cfg.PRED_THRESHOLD}
GT Threshold: {cfg.GT_THRESHOLD}
==================================================

[RED / POSITIVE CHANGES]
Total Objects -> GT: {r_tp + r_fn} | Pred: {r_tp + r_fp}
Raw Counts    -> TP: {r_tp} | FP: {r_fp} | FN: {r_fn}
Metrics       -> Precision: {r_prec:.4f} | Recall: {r_rec:.4f} | F1-Score: {r_f1:.4f}

--------------------------------------------------
[GREEN / NEGATIVE CHANGES]
Total Objects -> GT: {g_tp + g_fn} | Pred: {g_tp + g_fp}
Raw Counts    -> TP: {g_tp} | FP: {g_fp} | FN: {g_fn}
Metrics       -> Precision: {g_prec:.4f} | Recall: {g_rec:.4f} | F1-Score: {g_f1:.4f}

==================================================
[OVERALL COMBINED PERFORMANCE]
Total Objects -> GT: {o_tp + o_fn} | Pred: {o_tp + o_fp}
Raw Counts    -> TP: {o_tp} | FP: {o_fp} | FN: {o_fn}
Metrics       -> Precision: {o_prec:.4f} | Recall: {o_rec:.4f} | F1-Score: {o_f1:.4f}
==================================================

--------------------------------------------------
[METRIC DEFINITIONS]
* True Positives (TP) : Predicted objects that matched a GT object (IoU >= {cfg.IOU_THRESHOLD}).
* False Positives (FP): Predicted objects that did NOT match any GT object.
* False Negatives (FN): GT objects completely missed by the prediction.

* Precision : TP / (TP + FP)
* Recall    : TP / (TP + FN)
* F1-Score  : 2 * (Precision * Recall) / (Precision + Recall)
=================================================="""


def run_cc_metrics():
    """Main orchestrator for object-level metrics report."""
    print(f"\n[INFO] Running Object-Level Dataset Counts on {cfg.DEVICE}...")

    model = load_active_model()
    test_loader, test_files = get_test_dataloader()

    r_counts, g_counts = _accumulate_object_counts(model, test_loader)
    report = _format_metrics_report(r_counts, g_counts, len(test_files))

    os.makedirs(os.path.dirname(cfg.EVAL_FILE_CC_METRICS), exist_ok=True)
    with open(cfg.EVAL_FILE_CC_METRICS, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"[SUCCESS] Saved to: {cfg.EVAL_FILE_CC_METRICS}")
