"""
Model save/load utilities.
Handles persisting trained weights and loading them for evaluation.
"""

import os
import torch

import config as cfg
from training.models import get_active_model


def save_model(model, directory, filename):
    """Saves model state_dict to a specified location."""
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model weights saved to: {save_path}")


def save_active_model(model):
    """Saves the model using paths defined in config.py."""
    save_model(model, cfg.MODEL_WEIGHTS_DIR, cfg.MODEL_WEIGHTS_FILENAME)


def load_active_model():
    """Initializes the active model and loads its trained weights."""
    if not os.path.exists(cfg.MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {cfg.MODEL_WEIGHTS_PATH}. Have you run trainer.py?")

    model = get_active_model()
    model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH, map_location=cfg.DEVICE, weights_only=True))
    model.to(cfg.DEVICE).eval()
    print(f"[INFO] Active model loaded successfully from {cfg.MODEL_WEIGHTS_PATH}")
    return model


def predict_heatmap(model, prior, current):
    """
    Given an AngleRegressionNet, prior DRR and current DRR:
    1. Predicts the 2D angle.
    2. Rotates the 2D prior DRR by the predicted angle.
    3. Subtracts the rotated prior from current to form the predicted heatmap.
    """
    from torchvision.transforms.functional import rotate
    import torchvision
    
    # 1. Predict angle
    predicted_angles = model(prior, current) # (B,)
    
    # 2. Rotate each prior in the batch
    rotated_priors = []
    for i in range(prior.size(0)):
        # torchvision rotate expects angle in degrees
        angle_deg = predicted_angles[i].item()
        
        # prior is (B, C, H, W). We rotate (C, H, W)
        rot_p = rotate(prior[i], angle_deg, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        rotated_priors.append(rot_p)
    
    rotated_priors = torch.stack(rotated_priors)
    
    # 3. Subtract
    heatmap = current - rotated_priors
    return heatmap
