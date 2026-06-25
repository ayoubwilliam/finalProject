"""
Training loop with mixed-precision (AMP) support.
Pipeline: seed → build model → iterate epochs → save weights.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config as cfg
from training.models import get_active_model
from training.model_io import save_active_model
from training.data_loader import get_train_dataloader


def set_seed():
    """Locks RNG seeds for reproducibility."""
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)


def train_one_batch(model, prior, current, target, optimizer, criterion, scaler):
    """Executes a single forward/backward pass with AMP."""
    prior, current, target = prior.to(cfg.DEVICE), current.to(cfg.DEVICE), target.to(cfg.DEVICE)
    optimizer.zero_grad()
    with autocast():
        loss = criterion(model(prior, current), target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def train():
    """Main training loop."""
    print(f"Running on {cfg.DEVICE} | Model: {cfg.SELECTED_MODEL}")
    set_seed()

    train_loader = get_train_dataloader()
    model = get_active_model().to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion, scaler = nn.MSELoss(), GradScaler()

    pbar = tqdm(total=len(train_loader) * cfg.EPOCHS, desc="Training Progress", unit="batch")

    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        for prior, current, target, _ in train_loader:
            loss = train_one_batch(model, prior, current, target, optimizer, criterion, scaler)
            running_loss += loss
            pbar.update(1)
            pbar.set_postfix(epoch=f"{epoch + 1}/{cfg.EPOCHS}", loss=f"{loss:.5f}")
        pbar.write(f"Epoch {epoch + 1} Complete - Avg Loss: {running_loss / len(train_loader):.5f}")

    pbar.close()
    save_active_model(model)
    print("[INFO] Training Script Complete.")


if __name__ == "__main__":
    train()
