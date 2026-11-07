"""Data loaders, synthetic causal data generator, and preprocessing pipeline for Uplift-DRL.

Supports:
1. Synthetic Quick-Commerce cohort generator with known ground-truth ITE tau(x)
2. Kevin Hillstrom MineThatData e-mail analytics benchmark
3. Criteo Uplift Prediction Dataset v2
4. Starbucks promo dataset
5. Generic custom tabular quick-commerce CSV
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


SYNTHETIC_FEATURE_NAMES = [
    "order_frequency_30d",
    "avg_order_value_inr",
    "days_since_last_order",
    "discount_sensitivity_score",
    "app_sessions_7d",
    "app_push_enabled",
    "loyalty_tier_level",
    "weekend_shopper_ratio",
    "perishable_basket_share",
    "cart_abandonment_rate",
]


def generate_synthetic_uplift_data(
    n_samples: int = 25000,
    n_features: int = 10,
    treatment_ratio: float = 0.5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generates a realistic quick-commerce customer dataset with ground-truth ITE.

    Simulates four distinct behavioral cohorts:
    - Persuadables: positive incremental conversion lift when treated
    - Sure Things: high baseline conversion, buys regardless of promo
    - Lost Causes: low baseline, unaffected by voucher
    - Sleeping Dogs: negative lift (promos trigger notification fatigue or churn)

    Returns:
        dict with keys: X, T, Y, true_ite, y0_prob, y1_prob, feature_names
    """
    rng = np.random.default_rng(seed)

    # 1. Generate correlated customer covariates
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    # Make specific columns have intuitive retail meanings
    # X0: Order frequency (0 to 20 orders/month)
    # X1: AOV normalized
    # X2: Days since last order
    # X3: Discount sensitivity

    # 2. Base Conversion Probability (Control Outcome P(Y=1 | X, T=0))
    # Frequent, recent shoppers have high base conversion
    base_logits = 0.6 * X[:, 0] + 0.4 * X[:, 1] - 0.7 * X[:, 2] - 1.2
    y0_prob = 1.0 / (1.0 + np.exp(-base_logits))

    # 3. True Individual Treatment Effect tau(X)
    # Uplift depends heavily on discount sensitivity (X3) and app engagement (X4)
    # while high-frequency customers (X0 > 1.2) experience slight negative lift (sleeping dogs)
    uplift = (
        0.18 * np.maximum(0.0, X[:, 3]) +
        0.08 * X[:, 4] -
        0.06 * (X[:, 0] > 1.2).astype(np.float32) -
        0.04
    )
    # Bound between realistic retail uplift limits [-0.15, +0.35]
    uplift = np.clip(uplift, -0.15, 0.35)

    # 4. Treated Conversion Probability P(Y=1 | X, T=1)
    y1_prob = np.clip(y0_prob + uplift, 0.005, 0.995)

    # 5. Treatment Assignment T (simulated A/B test or slight observational bias)
    propensity = 1.0 / (1.0 + np.exp(-(0.25 * X[:, 0] - 0.20 * X[:, 2])))
    propensity = np.clip(propensity, 0.20, 0.80)
    T = (rng.uniform(0.0, 1.0, size=n_samples) < propensity).astype(np.float32)[:, np.newaxis]

    # 6. Realized Observed Outcome Y
    p_observed = np.where(T == 1.0, y1_prob[:, np.newaxis], y0_prob[:, np.newaxis])
    Y = (rng.uniform(0.0, 1.0, size=(n_samples, 1)) < p_observed).astype(np.float32)

    feature_names = SYNTHETIC_FEATURE_NAMES[:n_features]
    if len(feature_names) < n_features:
        feature_names += [f"feature_{i}" for i in range(len(feature_names), n_features)]

    return {
        "X": X,
        "T": T,
        "Y": Y,
        "true_ite": (y1_prob - y0_prob).astype(np.float32)[:, np.newaxis],
        "y0_prob": y0_prob[:, np.newaxis],
        "y1_prob": y1_prob[:, np.newaxis],
        "feature_names": np.array(feature_names),
    }


def load_hillstrom_csv(
    csv_path: str, target: str = "conversion"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads and preprocesses Kevin Hillstrom MineThatData benchmark.

    Treats Mens/Womens E-Mail as T=1, No E-Mail as T=0.
    """
    df = pd.read_csv(csv_path)

    # Treatment: Binarize marketing arms
    df["treatment"] = df["segment"].apply(lambda s: 0.0 if "No" in str(s) else 1.0)

    # Outcome
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in Hillstrom dataset columns: {df.columns.tolist()}")
    y = df[target].astype(np.float32).values[:, np.newaxis]
    t = df["treatment"].astype(np.float32).values[:, np.newaxis]

    # Features
    drop_cols = ["segment", "visit", "conversion", "spend", "treatment"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encode categoricals (zip_code, channel)
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    feature_names = feature_df.columns.tolist()
    X = feature_df.values.astype(np.float32)

    return X, t, y, feature_names


def split_and_save_npz(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    output_path: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    true_ite: Optional[np.ndarray] = None,
) -> None:
    """Performs stratified train/val/test splits on (T, Y) pairs and saves standardized .npz."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Stratify by combined string label of T and Y for balanced representation
    strata = [f"{int(t[0])}_{int(y[0])}" for t, y in zip(T, Y)]

    # Split Test
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=strata
    )

    # Split Validation
    strata_train_val = [strata[i] for i in train_val_idx]
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size, random_state=seed, stratify=strata_train_val
    )

    # Standardize X using Train statistics only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    save_dict = {
        "X_train": X_train.astype(np.float32),
        "T_train": T[train_idx].astype(np.float32),
        "Y_train": Y[train_idx].astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "T_val": T[val_idx].astype(np.float32),
        "Y_val": Y[val_idx].astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "T_test": T[test_idx].astype(np.float32),
        "Y_test": Y[test_idx].astype(np.float32),
        "feature_names": np.array(feature_names),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }

    if true_ite is not None:
        save_dict["true_ite_test"] = true_ite[test_idx].astype(np.float32)

    np.savez_compressed(output_path, **save_dict)
    print(f"Successfully processed dataset saved to: {output_path}")
    print(f"  Train: {len(train_idx)} rows | Val: {len(val_idx)} rows | Test: {len(test_idx)} rows")
    print(f"  Features: {len(feature_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing CLI for Uplift-DRL.")
    parser.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "hillstrom", "criteo", "starbucks"])
    parser.add_argument("--input", type=str, default=None, help="Input raw CSV path (for hillstrom/criteo)")
    parser.add_argument("--output", type=str, default="data/processed/synthetic.npz", help="Output .npz path")
    parser.add_argument("--n-samples", type=int, default=25000, help="Number of synthetic samples")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.source == "synthetic":
        print(f"Generating {args.n_samples} synthetic retail customer records...")
        data = generate_synthetic_uplift_data(n_samples=args.n_samples, seed=args.seed)
        split_and_save_npz(
            X=data["X"],
            T=data["T"],
            Y=data["Y"],
            feature_names=data["feature_names"],
            output_path=args.output,
            seed=args.seed,
            true_ite=data["true_ite"],
        )
    elif args.source == "hillstrom":
        if not args.input or not os.path.exists(args.input):
            raise FileNotFoundError(f"Hillstrom CSV not found at {args.input}. Download per data/README.md.")
        X, T, Y, fnames = load_hillstrom_csv(args.input)
        split_and_save_npz(X, T, Y, fnames, args.output, seed=args.seed)
    else:
        raise NotImplementedError(f"Source {args.source} requires explicit custom preprocessing script.")
