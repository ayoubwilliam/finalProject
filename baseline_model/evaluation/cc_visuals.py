"""
Connected components visual debugging grids.
Generates 5-row diagnostic layouts showing: full overlays, channel separation,
labeled CCs, pixel overlaps, and TP/FP/FN object masks.
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import matplotlib.patches as mpatches
from tqdm import tqdm

import config as cfg
from training.model_io import load_active_model, predict_heatmap
from training.data_loader import get_test_dataloader
from evaluation import cc_lib
from lib.overlay_heatmap import build_overlay_colormap, plot_heatmap_overlay


# ==============================================================================
# HELPER MASK GENERATORS
# ==============================================================================

def _create_rgb_mask_image(tp_mask, fp_mask, fn_mask):
    """Creates a dark-gray RGB image with TP (Green), FP (Red), FN (Yellow)."""
    H, W = tp_mask.shape
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    img[tp_mask] = [0, 255, 0]
    img[fp_mask] = [255, 0, 0]
    img[fn_mask] = [255, 255, 0]
    return img


def _create_pixel_overlap_image(gt_mask, pred_mask):
    """Creates an RGB image mapping pixel-level overlaps."""
    H, W = gt_mask.shape
    overlap_img = np.full((H, W, 3), 30, dtype=np.uint8)
    overlap_img[(gt_mask == 1) & (pred_mask == 1)] = [0, 255, 0]
    overlap_img[(gt_mask == 0) & (pred_mask == 1)] = [255, 0, 0]
    overlap_img[(gt_mask == 1) & (pred_mask == 0)] = [255, 255, 0]
    return overlap_img


# ==============================================================================
# ROW PLOTTING HELPERS
# ==============================================================================

def _plot_row1_full_overlays(fig, gs, current_np, target_np, pred_np, cmap):
    """Row 1: Full GT and Full Prediction overlays."""
    plot_heatmap_overlay(fig.add_subplot(gs[0, 0:2]), current_np, target_np, "Full GT", cmap, fig)
    plot_heatmap_overlay(fig.add_subplot(gs[0, 2:4]), current_np, pred_np, "Full Prediction", cmap, fig)


def _plot_row2_channel_separation(fig, gs, current_np, target_np, pred_np, analysis, cmap):
    """Row 2: Isolated red and green channel overlays."""
    plot_heatmap_overlay(fig.add_subplot(gs[1, 0]), current_np, target_np * analysis.gt_red_bin, "GT (Red Only)", cmap, fig)
    plot_heatmap_overlay(fig.add_subplot(gs[1, 1]), current_np, pred_np * analysis.pred_red_bin, "Prediction (Red Only)", cmap, fig)
    plot_heatmap_overlay(fig.add_subplot(gs[1, 2]), current_np, target_np * analysis.gt_green_bin, "GT (Green Only)", cmap, fig)
    plot_heatmap_overlay(fig.add_subplot(gs[1, 3]), current_np, pred_np * analysis.pred_green_bin, "Prediction (Green Only)", cmap, fig)


def _plot_row3_connected_components(fig, gs, analysis, cmap):
    """Row 3: Discrete connected components with unique colors."""
    plots = [
        (gs[2, 0], analysis.gt_red_labeled, analysis.gt_red_num, "GT RED CCs"),
        (gs[2, 1], analysis.pred_red_labeled, analysis.pred_red_num, "Pred RED CCs"),
        (gs[2, 2], analysis.gt_green_labeled, analysis.gt_green_num, "GT GREEN CCs"),
        (gs[2, 3], analysis.pred_green_labeled, analysis.pred_green_num, "Pred GREEN CCs"),
    ]

    for pos, label_img, num_items, title in plots:
        ax = fig.add_subplot(pos)
        ax.imshow(np.ma.masked_where(label_img == 0, label_img), cmap=cmap, vmin=1, vmax=max(num_items, 2))
        ax.axis('off')
        ax.set_title(title, fontsize=14)


def _plot_row4_pixel_overlaps(fig, gs, analysis):
    """Row 4: Raw pixel-level overlap masks."""
    ax_r1 = fig.add_subplot(gs[3, 0:2])
    ax_r1.imshow(_create_pixel_overlap_image(analysis.gt_red_bin, analysis.pred_red_bin))
    ax_r1.axis('off')
    ax_r1.set_title("RED Pixel Overlap", fontsize=14)

    ax_g1 = fig.add_subplot(gs[3, 2:4])
    ax_g1.imshow(_create_pixel_overlap_image(analysis.gt_green_bin, analysis.pred_green_bin))
    ax_g1.axis('off')
    ax_g1.set_title("GREEN Pixel Overlap", fontsize=14)


def _plot_row5_object_evaluation(fig, gs, analysis):
    """Row 5: Object-level TP/FP/FN masks."""
    ax_r2 = fig.add_subplot(gs[4, 0:2])
    ax_r2.imshow(_create_rgb_mask_image(analysis.r_tp_mask, analysis.r_fp_mask, analysis.r_fn_mask))
    ax_r2.axis('off')
    ax_r2.set_title(f"RED Object Eval (IoU > {cfg.IOU_THRESHOLD})", fontsize=14)

    ax_g2 = fig.add_subplot(gs[4, 2:4])
    ax_g2.imshow(_create_rgb_mask_image(analysis.g_tp_mask, analysis.g_fp_mask, analysis.g_fn_mask))
    ax_g2.axis('off')
    ax_g2.set_title(f"GREEN Object Eval (IoU > {cfg.IOU_THRESHOLD})", fontsize=14)


def _add_figure_decorations(fig):
    """Adds side labels, legends, and separating lines."""
    labels = [
        (0.82, "1) GT vs Prediction"),
        (0.66, "2) Red & Green Separation"),
        (0.50, "3) Found CCs"),
        (0.33, "4) Overlap Map"),
        (0.16, "5) TP/FP/FN"),
    ]
    for y, text in labels:
        fig.text(0.01, y, text, rotation=90, va='center', ha='center', fontsize=16, fontweight='bold')

    tp_patch = mpatches.Patch(color='#00FF00', label='Green: True Overlap (TP)')
    fp_patch = mpatches.Patch(color='#FF0000', label='Red: False Selection (FP)')
    fn_patch = mpatches.Patch(color='#FFFF00', label='Yellow: Missed Selection (FN)')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower center', ncol=3, fontsize=16,
               bbox_to_anchor=(0.5, 0.02))

    fig.add_artist(lines.Line2D([0.5, 0.5], [0.08, 0.76], transform=fig.transFigure, color="black", linewidth=5))
    fig.add_artist(lines.Line2D([0.5, 0.5], [0.08, 0.76], transform=fig.transFigure, color="gray", linewidth=2))


# ==============================================================================
# MAIN VISUALIZATION BUILDER
# ==============================================================================

def _save_cc_visualization(current_np, target_np, pred_np, analysis, file_id, save_path):
    """Constructs the 5-row GridSpec layout and saves the diagnostic image."""
    fig = plt.figure(figsize=cfg.PLOT_FIGSIZE_CC)

    title_text = (f"Master CC Analysis | Sample: {file_id}\n"
                  f"RED Objects -> TP: {analysis.r_tp} | FP: {analysis.r_fp} | FN: {analysis.r_fn}      "
                  f"GREEN Objects -> TP: {analysis.g_tp} | FP: {analysis.g_fp} | FN: {analysis.g_fn}")
    fig.suptitle(title_text, fontsize=22, fontweight='bold')

    gs = gridspec.GridSpec(5, 4, figure=fig, height_ratios=[1.2, 1, 1, 1, 1], left=0.04, bottom=0.08)
    overlay_cmap = build_overlay_colormap()
    cc_cmap = plt.cm.get_cmap('rainbow').copy()
    cc_cmap.set_bad(color='black')

    _plot_row1_full_overlays(fig, gs, current_np, target_np, pred_np, overlay_cmap)
    _plot_row2_channel_separation(fig, gs, current_np, target_np, pred_np, analysis, overlay_cmap)
    _plot_row3_connected_components(fig, gs, analysis, cc_cmap)
    _plot_row4_pixel_overlaps(fig, gs, analysis)
    _plot_row5_object_evaluation(fig, gs, analysis)

    _add_figure_decorations(fig)

    plt.subplots_adjust(top=0.94, hspace=0.3)
    plt.savefig(save_path, dpi=cfg.PLOT_DPI_HIGH)
    plt.close(fig)


def run_cc_visuals():
    """Generates diagnostic grids for a random subset of test files."""
    print(f"\n[INFO] Running DIVERSE Sample CC Visualizer on {cfg.DEVICE}...")
    os.makedirs(cfg.EVAL_CC_VISUALS_DIR, exist_ok=True)

    model = load_active_model()
    test_loader, test_files = get_test_dataloader()

    random.seed(cfg.SEED)
    selected_files = set(random.sample(test_files, min(cfg.CC_VISUAL_MAX_SAMPLES, len(test_files))))
    print(f"[INFO] Evaluating a diverse subset of {len(selected_files)} samples.")

    with torch.no_grad():
        for prior_b, current_b, target_angle, filenames, target_b in tqdm(test_loader, desc="Generating CC Overlays"):
            pred_b = predict_heatmap(model, prior_b.to(cfg.DEVICE), current_b.to(cfg.DEVICE))

            for i in range(prior_b.shape[0]):
                file_id = filenames[i]
                if file_id not in selected_files:
                    continue

                sanitized_id = file_id.replace(os.path.sep, '_')
                save_path = os.path.join(cfg.EVAL_CC_VISUALS_DIR, f"CC_Sample_{sanitized_id}.png")

                pred_sq = pred_b[i].squeeze()
                tgt_sq = target_b[i].squeeze().to(cfg.DEVICE)

                analysis = cc_lib.analyze_single_sample(
                    pred_sq, tgt_sq, cfg.PRED_THRESHOLD, cfg.GT_THRESHOLD, cfg.IOU_THRESHOLD,
                )

                _save_cc_visualization(
                    current_b[i].squeeze().cpu().numpy(),
                    tgt_sq.cpu().numpy(),
                    pred_sq.cpu().numpy(),
                    analysis, file_id, save_path,
                )
