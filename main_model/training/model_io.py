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
