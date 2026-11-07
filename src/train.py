"""Unified Training Pipeline for Uplift-DRL Estimators.

Supports CLI model switching:
    python -m src.train --model dragonnet --data data/processed/synthetic.npz
    python -m src.train --model tarnet    --data data/processed/synthetic.npz
    python -m src.train --model t         --data data/processed/synthetic.npz
    python -m src.train --model s         --data data/processed/synthetic.npz

Tracks epoch loss, validates on Qini coefficient, and performs early stopping.
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval import qini_curve
from src.losses import DragonNetLoss, StandardUpliftLoss, TARNetLoss
from src.models import DragonNet, SLearner, TARNet, TLearner, build_model
from src.utils import get_device, load_config, save_checkpoint, set_seed


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    model_type: str,
    device: torch.device,
) -> float:
    """Runs a single training epoch across mini-batches."""
    model.train()
    total_loss = 0.0

    for x_b, t_b, y_b in loader:
        x_b = x_b.to(device)
        t_b = t_b.to(device)
        y_b = y_b.to(device)

        optimizer.zero_grad()

        if model_type == "dragonnet":
            y0, y1, prop, eps = model(x_b)
            loss, _ = criterion(y_b, t_b, y0, y1, prop, eps)
        elif model_type == "tarnet":
            y0, y1 = model(x_b)
            loss, _ = criterion(y_b, t_b, y0, y1)
        elif model_type in ["s", "slearner"]:
            y_pred = model(x_b, t_b)
            loss, _ = criterion(y_pred, y_b)
        elif model_type in ["t", "tlearner"]:
            y_pred = model(x_b, t_b)
            loss, _ = criterion(y_pred, y_b)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * len(x_b)

    return total_loss / len(loader.dataset)


def evaluate(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    t_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates validation Qini coefficient and AUUC."""
    model.eval()
    with torch.no_grad():
        x_dev = x_val.to(device)
        uplift_preds = model.predict_uplift(x_dev).cpu().numpy().ravel()

    _, _, _, qini = qini_curve(y_val, t_val, uplift_preds)
    return float(qini), uplift_preds


def run_training(
    model_type: str,
    data_path: str,
    config_path: str = "configs/uplift.yaml",
    output_dir: str = "checkpoints",
    epochs_override: int = None,
    lr_override: float = None,
    batch_size_override: int = None,
) -> str:
    """Coordinates data loading, model compilation, training loop, and model checkpointing."""
    config = load_config(config_path)
    set_seed(config.get("train", {}).get("seed", 42))
    device = get_device()

    # Load preprocessed npz dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at: {data_path}. Run src.data first.")

    data = np.load(data_path)
    X_train = data["X_train"]
    T_train = data["T_train"]
    Y_train = data["Y_train"]

    X_val = data["X_val"]
    T_val = data["T_val"]
    Y_val = data["Y_val"]

    feature_names = data.get("feature_names", np.array([f"f_{i}" for i in range(X_train.shape[1])]))

    # Hyperparameters
    train_cfg = config.get("train", {})
    batch_size = batch_size_override or train_cfg.get("batch_size", 512)
    lr = lr_override or float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = epochs_override or train_cfg.get("epochs", 50)
    patience = train_cfg.get("early_stopping_patience", 7)
    val_metric_name = train_cfg.get("val_metric", "qini")

    # DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(T_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model instantiation
    m_cfg = config.get("model", {})
    input_dim = X_train.shape[1]
    hidden_dim = m_cfg.get("hidden_dim", 128)
    n_layers = m_cfg.get("n_layers", 3)
    dropout = m_cfg.get("dropout", 0.2)

    normalized_model_type = model_type.lower().strip()
    model = build_model(
        model_type=normalized_model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    # Criterion
    if normalized_model_type in ["dragonnet", "dragon"]:
        d_cfg = config.get("dragonnet", {})
        criterion = DragonNetLoss(
            alpha=float(d_cfg.get("alpha", 1.0)),
            beta=float(d_cfg.get("beta", 1.0)),
        )
    elif normalized_model_type in ["tarnet"]:
        criterion = TARNetLoss()
    else:
        criterion = StandardUpliftLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training state
    best_val_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(output_dir, f"{normalized_model_type}_best.pt")

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32)

    print("\n" + "=" * 65)
    print(f" Uplift-DRL Training: {normalized_model_type.upper()}")
    print("=" * 65)
    print(f"  Device: {device} | Features: {input_dim} | Batch Size: {batch_size} | LR: {lr}")
    print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    print("-" * 65)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            model_type=normalized_model_type,
            device=device,
        )

        val_qini, _ = evaluate(model, val_tensor_x, T_val, Y_val, device)

        is_best = val_qini > best_val_score
        marker = "⭐ (Best)" if is_best else ""

        print(f" Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Qini: {val_qini:.4f} {marker}")

        if is_best:
            best_val_score = val_qini
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_type": normalized_model_type,
                    "model_state_dict": model.state_dict(),
                    "best_val_qini": best_val_score,
                    "config": config,
                    "input_dim": input_dim,
                    "feature_names": feature_names,
                    "scaler_mean": data.get("scaler_mean"),
                    "scaler_scale": data.get("scaler_scale"),
                },
                best_model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"✅ Training completed in {elapsed:.1f}s!")
    print(f"🏆 Best Validation Qini: {best_val_score:.4f}")
    print(f"💾 Checkpoint saved to: {best_model_path}")
    print("=" * 65 + "\n")

    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep causal uplift model.")
    parser.add_argument("--model", type=str, default="dragonnet", choices=["s", "t", "tarnet", "dragonnet"], help="Model architecture")
    parser.add_argument("--data", type=str, default="data/processed/synthetic.npz", help="Path to preprocessed .npz data")
    parser.add_argument("--config", type=str, default="configs/uplift.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    args = parser.parse_args()

    run_training(
        model_type=args.model,
        data_path=args.data,
        config_path=args.config,
        output_dir=args.output_dir,
        epochs_override=args.epochs,
        lr_override=args.lr,
        batch_size_override=args.batch_size,
    )

