"""
Module: run_consolidation.py
Provides functionality for run_consolidation.
"""

import os
import sys
import json
import re
import textwrap
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to sys.path to import config, models, lib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models and visualization logic
from models.architecture import VGGDiffNet
from lib.overlay_heatmap import build_overlay_colormap, plot_base_image, plot_heatmap_overlay
from preprocessing import preprocess_real_image
import config as cfg

# ================== CONFIGURATION ==================
CONSOLIDATION_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "consolidation_weights.pth")
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "consolidation_tube_pairs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
JSON_NAME = "clinical_changes.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Regex pattern to catch variations of keywords
KEYWORD_PATTERN = re.compile(r'\b(consolidations?|intubat(?:ion|ed|ing)|et tube|nasogastric tube|tubes?)\b', re.IGNORECASE)

# ===================================================

def highlight_keywords_console(text):
    """Highlights keywords in red for the terminal output."""
    return KEYWORD_PATTERN.sub(lambda m: f"\033[91m{m.group(1).upper()}\033[0m", text)


def highlight_keywords_txt(text):
    """Wraps keywords in triple dollar signs for the saved text file."""
    return KEYWORD_PATTERN.sub(lambda m: f"$$${m.group(1).upper()}$$$", text)


def process_patient(patient_id, model, rcg_cmap, model_type):
    """
    Runs the inference pipeline for a single patient using a specific model type.
    """
    data_dir = os.path.join(BASE_DATA_DIR, patient_id)
    json_path = os.path.join(data_dir, JSON_NAME)

    # 1. Scan for images and allow .jpg, .jpeg, and .png extensions
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(data_dir) if f.startswith("visit_") and f.lower().endswith(valid_exts)]

    # Sort files chronologically by their embedded index
    image_files.sort(key=lambda x: int(x.split('_')[1]))

    num_visits = len(image_files)
    if num_visits < 2:
        print(f"[{patient_id}] Skipped: Not enough images ({num_visits} found).")
        return

    # 2. Load the JSON clinical notes
    clinical_data = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            clinical_data = json.load(f)

    num_transitions = num_visits - 1

    # Initialize visual grid layout
    fig, axes = plt.subplots(nrows=num_transitions, ncols=3, figsize=(16, 5.5 * num_transitions))
    if num_transitions == 1:
        axes = np.expand_dims(axes, axis=0)

    # Initialize text file content
    txt_output_lines = []
    txt_output_lines.append(f"{'=' * 60}")
    txt_output_lines.append(f"PATIENT {patient_id} - CLINICAL GROUND TRUTH ({model_type.upper()} MODEL)")
    txt_output_lines.append(f"{'=' * 60}\n")

    print(f"\n{'=' * 60}")
    print(f"PATIENT {patient_id} - CLINICAL GROUND TRUTH [{model_type.upper()}]")
    print(f"{'=' * 60}\n")

    # 3. Iterate through consecutive pairs
    from PIL import Image
    for i in range(num_transitions):
        prior_file = image_files[i]
        current_file = image_files[i + 1]
        
        prior_path = os.path.join(data_dir, prior_file)
        current_path = os.path.join(data_dir, current_file)

        # Get original image dimensions for the accumulator
        orig_img = Image.open(current_path)
        W, H = orig_img.size
        accumulator_heatmap = np.zeros((H, W), dtype=np.float32)

        # Iterate over crop settings
        for trim, frac in cfg.CONSOLIDATION_CROP_SETTINGS:
            for dy, dx in cfg.TRANSLATION_SETTINGS:
                prior_tensor, _, _ = preprocess_real_image(prior_path, trim, frac, dy, dx)
                current_tensor, _, bbox_current = preprocess_real_image(current_path, trim, frac, dy, dx)

                with torch.no_grad():
                    output_tensor = model(prior_tensor.to(DEVICE), current_tensor.to(DEVICE))
                    
                    # Calculate bbox dimensions
                    y1, y2, x1, x2 = bbox_current
                    crop_h = y2 - y1
                    crop_w = x2 - x1

                    # Resize heatmap back to its crop size using torch
                    heatmap_resized = torch.nn.functional.interpolate(
                        output_tensor, size=(crop_h, crop_w), mode='bilinear', align_corners=False
                    ).squeeze().cpu().numpy()

                    # Update accumulator with the value that has the maximum absolute magnitude
                    current_acc = accumulator_heatmap[y1:y2, x1:x2]
                    mask = np.abs(heatmap_resized) > np.abs(current_acc)
                    accumulator_heatmap[y1:y2, x1:x2] = np.where(mask, heatmap_resized, current_acc)

        # Extract base visualization
        base_trim, base_frac = cfg.CONSOLIDATION_BASE_CROP
        base_dy, base_dx = cfg.BASE_TRANSLATION
        _, prior_np, _ = preprocess_real_image(prior_path, base_trim, base_frac, base_dy, base_dx)
        _, current_np, base_bbox = preprocess_real_image(current_path, base_trim, base_frac, base_dy, base_dx)

        # Extract final heatmap using base_bbox and resize back to 512x512
        y1, y2, x1, x2 = base_bbox
        extracted_heatmap = accumulator_heatmap[y1:y2, x1:x2]

        final_heatmap_tensor = torch.from_numpy(extracted_heatmap).unsqueeze(0).unsqueeze(0)
        heatmap_np = torch.nn.functional.interpolate(
            final_heatmap_tensor, size=(512, 512), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()

        # --- Apply Clipping ---
        # Set values below threshold to zero before plotting
        heatmap_np[np.abs(heatmap_np) < cfg.PRED_THRESHOLD] = 0

        # 4. Format headers
        transition_title = f"► TRANSITION: Visit {i} ➔ Visit {i + 1}"

        print(transition_title)
        print("-" * 60)

        txt_output_lines.append(transition_title)
        txt_output_lines.append("-" * 60)

        for visit_idx in (i, i + 1):
            if visit_idx < len(clinical_data):
                record = clinical_data[visit_idx]
                visit_header = f"[Visit {visit_idx} - Day {record.get('offset', '?')}]"
                intu_status = f"  • Intubated: {record.get('intubated', 'unknown')}"

                print(visit_header)
                print(intu_status)

                txt_output_lines.append(visit_header)
                txt_output_lines.append(intu_status)

                notes = record.get('clinical_notes', 'none')
                wrapped_notes = textwrap.fill(notes, width=80, subsequent_indent="    ")

                # Format for console (Red ANSI)
                console_notes = highlight_keywords_console(wrapped_notes)
                print(f"  • Notes: {console_notes}\n")

                # Format for text file ($$$$$$)
                txt_notes = highlight_keywords_txt(wrapped_notes)
                txt_output_lines.append(f"  • Notes: {txt_notes}\n")

        # 5. Populate the 3-panel visual grid
        plot_base_image(axes[i, 0], prior_np, "")
        plot_base_image(axes[i, 1], current_np, "")
        plot_heatmap_overlay(axes[i, 2], current_np, heatmap_np, "", rcg_cmap, fig)

    plt.tight_layout()

    # Define dedicated output directory for this model type
    model_output_dir = os.path.join(OUTPUT_DIR, model_type)
    os.makedirs(model_output_dir, exist_ok=True)

    # Save the output files into the respective subfolder
    img_filepath = os.path.join(model_output_dir, f"patient_{patient_id}_{model_type}_output.png")
    txt_filepath = os.path.join(model_output_dir, f"patient_{patient_id}_{model_type}_output.txt")

    # Save image
    plt.savefig(img_filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)

    # Save text
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_output_lines))

    print(f"[*] Saved {model_type} visual grid to: {img_filepath}")
    print(f"[*] Saved {model_type} clinical notes to: {txt_filepath}")


def run_batch_inference():
    """Functionality for run_batch_inference."""
    if not os.path.exists(BASE_DATA_DIR):
        print(f"Error: Base directory '{BASE_DATA_DIR}' not found.")
        return

    print(f"Loading model weights into {DEVICE}...")

    # 1. Load Consolidation Model
    consolidation_model = VGGDiffNet()
    consolidation_model.load_state_dict(torch.load(CONSOLIDATION_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    consolidation_model.to(DEVICE).eval()

    rcg_cmap = build_overlay_colormap()

    # 3. Find all patient folders in the base directory
    patient_dirs = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]

    print(f"Found {len(patient_dirs)} patient records to process.")

    # 4. Process each patient
    for patient_id in patient_dirs:
        # Run Consolidation Inference
        process_patient(patient_id, consolidation_model, rcg_cmap, model_type="consolidation")

    print(f"\n{'=' * 60}")
    print(f"BATCH PROCESSING COMPLETE. Outputs saved in '{OUTPUT_DIR}/consolidation'.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_batch_inference()
