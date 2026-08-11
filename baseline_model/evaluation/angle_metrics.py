import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg
from training.model_io import load_active_model
from training.data_loader import get_test_dataloader

def run_angle_metrics():
    """Calculates angle error and generates a scatter plot."""
    print(f"\n[INFO] Starting Angle Metrics Evaluation...")
    model = load_active_model()
    test_loader, _ = get_test_dataloader()
    
    errors = []
    
    with torch.no_grad():
        for prior, current, target_angle, filenames, _ in tqdm(test_loader, desc="Calculating Angle Error"):
            predicted_angles = model(prior.to(cfg.DEVICE), current.to(cfg.DEVICE)).cpu().numpy()
            target_angles = target_angle.numpy()
            
            for i in range(len(predicted_angles)):
                error = abs(predicted_angles[i] - target_angles[i])
                errors.append(error)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"[INFO] Mean Angle Error: {mean_error:.4f} degrees (Std: {std_error:.4f})")
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(errors)), errors, color='blue', alpha=0.6, label='Angle Error')
    plt.axhline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error ({mean_error:.2f})')
    
    plt.title("2D Angle Prediction Error per Test Sample")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Absolute Angle Error (degrees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(cfg.EVALUATION_DIR, exist_ok=True)
    plot_path = os.path.join(cfg.EVALUATION_DIR, "angle_error_plot.png")
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI_STANDARD)
    plt.close()
    
    print(f"[SUCCESS] Angle error plot saved to {plot_path}")
