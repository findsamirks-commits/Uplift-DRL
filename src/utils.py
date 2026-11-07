"""Utility functions for Uplift-DRL."""

import os
import random
from typing import Any, Dict
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Selects available CUDA GPU or defaults to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """Saves a model checkpoint dictionary."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, map_location: Any = None) -> Dict[str, Any]:
    """Loads a saved checkpoint dictionary."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    if map_location is None:
        map_location = get_device()
    return torch.load(filepath, map_location=map_location, weights_only=False)
