"""Batch inference and scoring pipeline for Uplift-DRL.

Loads trained checkpoints and computes individual treatment effect (CATE) estimates,
counterfactual conversion probabilities, and customer cohort classifications.
"""

import argparse
import os
import sys
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import DragonNet, build_model
from src.targeting_bridge import TargetingBridge
from src.utils import get_device, load_checkpoint


class UpliftPredictor:
    """Production batch scorer for trained Uplift-DRL models."""

    def __init__(self, model_path: str, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device) if device else get_device()
        self.checkpoint = load_checkpoint(model_path, map_location=self.device)

        self.model_type = self.checkpoint["model_type"]
        self.config = self.checkpoint.get("config", {})
        self.feature_names = self.checkpoint.get("feature_names", None)
        self.scaler_mean = self.checkpoint.get("scaler_mean", None)
        self.scaler_scale = self.checkpoint.get("scaler_scale", None)

        m_cfg = self.config.get("model", {})
        self.model = build_model(
            model_type=self.model_type,
            input_dim=self.checkpoint["input_dim"],
            hidden_dim=m_cfg.get("hidden_dim", 128),
            n_layers=m_cfg.get("n_layers", 3),
            dropout=0.0,
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        self.bridge = TargetingBridge()

    def _prepare_array(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Aligns columns and applies feature normalization."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            # If explicit feature_names stored, ensure alignment
            if self.feature_names is not None:
                avail_cols = [c for c in self.feature_names if c in df.columns]
                if len(avail_cols) == len(self.feature_names):
                    df = df[self.feature_names]
            arr = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)

        # Apply standard scaler if saved in checkpoint
        if self.scaler_mean is not None and self.scaler_scale is not None:
            arr = (arr - self.scaler_mean) / (self.scaler_scale + 1e-8)

        return arr

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Generates predictions and commercial cohort classifications.

        Returns DataFrame with:
            y0_prob, y1_prob, uplift_score, cohort, (and propensity if DragonNet)
        """
        arr = self._prepare_array(data)
        x_tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y0, y1 = self.model.predict_outcomes(x_tensor)
            y0_np = y0.cpu().numpy().ravel()
            y1_np = y1.cpu().numpy().ravel()
            uplift_np = y1_np - y0_np

            prop_np = None
            if isinstance(self.model, DragonNet):
                prop = self.model.predict_propensity(x_tensor)
                prop_np = prop.cpu().numpy().ravel()

        cohorts = self.bridge.classify_cohorts(uplift_scores=uplift_np, y0=y0_np, y1=y1_np)

        results = pd.DataFrame({
            "y0_control_prob": np.round(y0_np, 4),
            "y1_treated_prob": np.round(y1_np, 4),
            "uplift_score": np.round(uplift_np, 4),
            "cohort": cohorts,
        })

        if prop_np is not None:
            results["propensity_score"] = np.round(prop_np, 4)

        if isinstance(data, pd.DataFrame) and "customer_id" in data.columns:
            results.insert(0, "customer_id", data["customer_id"].values)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score customer dataset with trained uplift model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--input", type=str, required=True, help="Path to input scoring CSV")
    parser.add_argument("--output", type=str, default="uplift_scores.csv", help="Path to output CSV")

    args = parser.parse_args()

    predictor = UpliftPredictor(args.model_path)
    df_in = pd.read_csv(args.input)
    scored = predictor.predict(df_in)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored)} records. Output saved to: {args.output}")
    print("\nCohort distribution:")
    print(scored["cohort"].value_counts().to_string())
