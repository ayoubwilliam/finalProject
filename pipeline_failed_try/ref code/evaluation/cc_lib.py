"""
Connected components core library.
Data structures for analysis results and all CC logic: labeling, IoU matching,
TP/FP/FN classification, and instance-level Dice scoring.
"""

import numpy as np
from scipy.ndimage import label
from dataclasses import dataclass
import config as cfg


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class SampleAnalysisResult:
    """Container for all metrics and masks of a processed image (red + green channels)."""

    # --- RED CHANNEL (POSITIVE CHANGES) ---
    pred_red_bin: np.ndarray
    gt_red_bin: np.ndarray
    pred_red_labeled: np.ndarray
    gt_red_labeled: np.ndarray
    pred_red_num: int
    gt_red_num: int
    r_tp_mask: np.ndarray
    r_fp_mask: np.ndarray
    r_fn_mask: np.ndarray
    r_tp: int
    r_fp: int
    r_fn: int
    r_dice_tp_only: list

    # --- GREEN CHANNEL (NEGATIVE CHANGES) ---
    pred_green_bin: np.ndarray
    gt_green_bin: np.ndarray
    pred_green_labeled: np.ndarray
    gt_green_labeled: np.ndarray
    pred_green_num: int
    gt_green_num: int
    g_tp_mask: np.ndarray
    g_fp_mask: np.ndarray
    g_fn_mask: np.ndarray
    g_tp: int
    g_fp: int
    g_fn: int
    g_dice_tp_only: list


# ==============================================================================
# CORE UTILITIES
# ==============================================================================

def get_connected_components(binary_mask_np):
    """Finds distinct blobs in a binary mask and assigns unique integer IDs, filtering out small ones."""
    labeled_array, num_features = label(binary_mask_np)
    
    valid_features = 0
    new_labeled_array = np.zeros_like(labeled_array)
    for i in range(1, num_features + 1):
        mask = (labeled_array == i)
        if mask.sum() >= cfg.MIN_CC_SIZE:
            valid_features += 1
            new_labeled_array[mask] = valid_features
            
    return new_labeled_array, valid_features


def _calculate_iou(mask_a, mask_b):
    """Calculates Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def _calculate_dice(mask_a, mask_b):
    """Calculates pixel-wise Dice score between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    return (2.0 * intersection) / denom if denom > 0 else 0.0


def _find_best_match(p_mask, gt_labeled):
    """Scans all GT blobs touching the prediction to find the highest IoU match."""
    best_iou, best_gt = 0.0, -1
    touching_gts = np.unique(gt_labeled[p_mask])

    for g in touching_gts:
        if g == 0:
            continue
        g_mask = (gt_labeled == g)
        iou = _calculate_iou(p_mask, g_mask)
        if iou > best_iou:
            best_iou, best_gt = iou, g

    return best_iou, best_gt


def _binarize_channel(pred_sq, tgt_sq, pred_thresh, gt_thresh, is_positive):
    """Applies thresholds to extract purely positive or purely negative changes."""
    if is_positive:
        pred_bin = (pred_sq > pred_thresh).int().cpu().numpy()
        gt_bin = (tgt_sq > gt_thresh).int().cpu().numpy()
    else:
        pred_bin = (pred_sq < -pred_thresh).int().cpu().numpy()
        gt_bin = (tgt_sq < -gt_thresh).int().cpu().numpy()
    return pred_bin, gt_bin


# ==============================================================================
# CLASSIFICATION HELPERS
# ==============================================================================

def _is_true_positive(best_iou, iou_threshold):
    """A prediction is a TP if its best overlap meets the threshold."""
    return best_iou >= iou_threshold


def _is_false_positive(best_iou, iou_threshold):
    """A prediction is an FP if it fails to meet the overlap threshold."""
    return best_iou < iou_threshold


def _is_false_negative(gt_id, matched_gt_set):
    """A GT object is an FN if no prediction matched with it."""
    return gt_id not in matched_gt_set


# ==============================================================================
# EVALUATION LOGIC
# ==============================================================================

def calculate_object_metrics(gt_labeled, gt_num, pred_labeled, pred_num, iou_threshold):
    """Categorizes all blobs into TP, FP, FN. Returns visual masks and raw counts."""
    tp_pred_mask = np.zeros_like(pred_labeled, dtype=bool)
    fp_pred_mask = np.zeros_like(pred_labeled, dtype=bool)
    fn_gt_mask = np.zeros_like(gt_labeled, dtype=bool)

    tp_count, fp_count, fn_count = 0, 0, 0
    matched_gt, matched_pred = set(), set()

    # evaluate all predictions
    for p in range(1, pred_num + 1):
        p_mask = (pred_labeled == p)
        best_iou, best_gt = _find_best_match(p_mask, gt_labeled)

        if _is_true_positive(best_iou, iou_threshold):
            matched_pred.add(p)
            matched_gt.add(best_gt)
            tp_pred_mask |= p_mask
            tp_count += 1
        elif _is_false_positive(best_iou, iou_threshold):
            fp_pred_mask |= p_mask
            fp_count += 1

    # find missed ground truths
    for g in range(1, gt_num + 1):
        if _is_false_negative(g, matched_gt):
            fn_gt_mask |= (gt_labeled == g)
            fn_count += 1

    return tp_pred_mask, fp_pred_mask, fn_gt_mask, tp_count, fp_count, fn_count


def calculate_instance_dice_scores(gt_labeled, gt_num, pred_labeled, pred_num, iou_threshold):
    """Calculates Instance Dice scores for TP matches only."""
    dice_tp_only = []

    for p in range(1, pred_num + 1):
        p_mask = (pred_labeled == p)
        best_iou, best_gt = _find_best_match(p_mask, gt_labeled)

        if _is_true_positive(best_iou, iou_threshold) and best_gt != -1:
            g_mask = (gt_labeled == best_gt)
            dice = _calculate_dice(p_mask, g_mask)
            dice_tp_only.append(dice)

    # perfect empty evaluation edge-case
    if gt_num == 0 and pred_num == 0:
        dice_tp_only.append(1.0)

    return dice_tp_only


# ==============================================================================
# MASTER PIPELINE
# ==============================================================================

def _analyze_color_channel(pred_sq, tgt_sq, pred_thresh, gt_thresh, iou_thresh, is_positive):
    """Runs binarize → label → evaluate pipeline for a single color channel."""
    pred_bin, gt_bin = _binarize_channel(pred_sq, tgt_sq, pred_thresh, gt_thresh, is_positive)

    gt_labeled, gt_num = get_connected_components(gt_bin)
    pred_labeled, pred_num = get_connected_components(pred_bin)

    tp_m, fp_m, fn_m, tp, fp, fn = calculate_object_metrics(
        gt_labeled, gt_num, pred_labeled, pred_num, iou_thresh
    )

    dice_tp = calculate_instance_dice_scores(
        gt_labeled, gt_num, pred_labeled, pred_num, iou_thresh
    )

    return (
        pred_bin, gt_bin, pred_labeled, gt_labeled, pred_num, gt_num,
        tp_m, fp_m, fn_m, tp, fp, fn, dice_tp
    )


def analyze_single_sample(pred_sq, tgt_sq, pred_thresh, gt_thresh, iou_thresh):
    """Analyzes both color channels and aggregates into a SampleAnalysisResult."""

    (r_p_bin, r_g_bin, r_p_lab, r_g_lab, r_p_num, r_g_num,
     r_tpm, r_fpm, r_fnm, r_tp, r_fp, r_fn, r_dice_tp) = _analyze_color_channel(
        pred_sq, tgt_sq, pred_thresh, gt_thresh, iou_thresh, is_positive=True
    )

    (g_p_bin, g_g_bin, g_p_lab, g_g_lab, g_p_num, g_g_num,
     g_tpm, g_fpm, g_fnm, g_tp, g_fp, g_fn, g_dice_tp) = _analyze_color_channel(
        pred_sq, tgt_sq, pred_thresh, gt_thresh, iou_thresh, is_positive=False
    )

    return SampleAnalysisResult(
        pred_red_bin=r_p_bin, gt_red_bin=r_g_bin,
        pred_red_labeled=r_p_lab, gt_red_labeled=r_g_lab,
        pred_red_num=r_p_num, gt_red_num=r_g_num,
        r_tp_mask=r_tpm, r_fp_mask=r_fpm, r_fn_mask=r_fnm,
        r_tp=r_tp, r_fp=r_fp, r_fn=r_fn,
        r_dice_tp_only=r_dice_tp,
        pred_green_bin=g_p_bin, gt_green_bin=g_g_bin,
        pred_green_labeled=g_p_lab, gt_green_labeled=g_g_lab,
        pred_green_num=g_p_num, gt_green_num=g_g_num,
        g_tp_mask=g_tpm, g_fp_mask=g_fpm, g_fn_mask=g_fnm,
        g_tp=g_tp, g_fp=g_fp, g_fn=g_fn,
        g_dice_tp_only=g_dice_tp,
    )
