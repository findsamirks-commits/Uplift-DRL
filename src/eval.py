"""Evaluation metrics and diagnostics for Uplift and CATE modeling.

Pure NumPy implementations of:
- Qini Curve and Qini Coefficient
- Area Under the Uplift Curve (AUUC)
- Uplift Decile Table with observed vs predicted lift
- Inverse Propensity Weighted (IPW) Expected Policy Value
"""

import argparse
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


def _to_numpy_flat(arr: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
    """Flattens input into a 1D float NumPy array."""
    if isinstance(arr, pd.Series):
        arr = arr.values
    return np.asarray(arr, dtype=np.float64).ravel()


def qini_curve(
    y_true: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    uplift_scores: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculates the Qini curve and Qini coefficient.

    Sorts candidates descending by predicted uplift.
    At rank k:
        Q(k) = Y_T(k) - Y_C(k) * (N_T(k) / N_C(k))

    Args:
        y_true: Binary ground-truth outcome (conversion/click, 0 or 1).
        treatment: Binary treatment indicator (1 for treated, 0 for control).
        uplift_scores: Predicted CATE / uplift scores.

    Returns:
        fractions: Array of targeted population fractions (0 to 1).
        qini_curve: Cumulative incremental conversions at each rank.
        random_baseline: Expected incremental conversions under random targeting.
        qini_score: Area between Qini curve and random baseline.
    """
    y = _to_numpy_flat(y_true)
    t = _to_numpy_flat(treatment)
    scores = _to_numpy_flat(uplift_scores)

    n = len(y)
    if n == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.0

    # Sort descending by predicted uplift (with secondary stable sort on random jitter)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    t_sorted = t[order]

    # Cumulative counts of treated and control
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1.0 - t_sorted)

    # Cumulative conversions
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1.0 - t_sorted))

    # Total counts
    total_t = cum_t[-1]
    total_c = cum_c[-1]

    # Avoid divide by zero
    ratio_t_c = np.zeros(n, dtype=np.float64)
    valid_c = cum_c > 0
    ratio_t_c[valid_c] = cum_t[valid_c] / cum_c[valid_c]

    # Qini value at each rank
    q_vals = cum_y_t - (cum_y_c * ratio_t_c)

    # Prepend origin (0, 0)
    fractions = np.linspace(0.0, 1.0, n + 1)
    qini_vals = np.concatenate(([0.0], q_vals))

    # Random baseline straight line from 0 to total final Qini
    total_ratio = (total_t / total_c) if total_c > 0 else 1.0
    final_q = cum_y_t[-1] - (cum_y_c[-1] * total_ratio)
    random_vals = fractions * final_q

    # Calculate area between model curve and random baseline using trapezoidal integration
    area_model = np.trapezoid(qini_vals, fractions) if hasattr(np, "trapezoid") else np.trapz(qini_vals, fractions)
    area_random = np.trapezoid(random_vals, fractions) if hasattr(np, "trapezoid") else np.trapz(random_vals, fractions)
    qini_score = float(area_model - area_random)

    return fractions, qini_vals, random_vals, qini_score


def auuc_score(
    y_true: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    uplift_scores: Union[np.ndarray, pd.Series],
) -> float:
    """Calculates the Area Under the Uplift Curve (AUUC).

    Uses cumulative conversion rate differences:
        U(k) = ( (Y_T(k) / N_T(k)) - (Y_C(k) / N_C(k)) ) * (N_T(k) + N_C(k))

    Args:
        y_true: Binary ground-truth outcome.
        treatment: Binary treatment indicator.
        uplift_scores: Predicted CATE scores.

    Returns:
        auuc: Integrated area under the cumulative uplift gain curve.
    """
    y = _to_numpy_flat(y_true)
    t = _to_numpy_flat(treatment)
    scores = _to_numpy_flat(uplift_scores)

    n = len(y)
    if n == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    t_sorted = t[order]

    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1.0 - t_sorted)
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1.0 - t_sorted))

    # Rates
    p_t = np.zeros(n, dtype=np.float64)
    p_c = np.zeros(n, dtype=np.float64)

    valid_t = cum_t > 0
    valid_c = cum_c > 0
    p_t[valid_t] = cum_y_t[valid_t] / cum_t[valid_t]
    p_c[valid_c] = cum_y_c[valid_c] / cum_c[valid_c]

    # Lift * cumulative size
    u_vals = (p_t - p_c) * (cum_t + cum_c)
    u_curve = np.concatenate(([0.0], u_vals))
    fractions = np.linspace(0.0, 1.0, n + 1)

    area = np.trapezoid(u_curve, fractions) if hasattr(np, "trapezoid") else np.trapz(u_curve, fractions)
    return float(area)


def uplift_decile_table(
    y_true: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    uplift_scores: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Partitions scored population into n_bins and calculates cohort response metrics.

    Decile 1 represents customers with the highest predicted uplift.

    Returns DataFrame with columns:
        decile, n_total, n_treated, n_control,
        treated_conv_rate, control_conv_rate, observed_uplift,
        predicted_uplift_mean, incremental_conversions
    """
    df = pd.DataFrame({
        "y": _to_numpy_flat(y_true),
        "t": _to_numpy_flat(treatment),
        "score": _to_numpy_flat(uplift_scores),
    })

    # Rank and bin (1 = highest score, 10 = lowest)
    df["rank"] = df["score"].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(df["rank"], q=n_bins, labels=range(1, n_bins + 1))

    rows = []
    for d, group in df.groupby("decile", observed=False):
        n_tot = len(group)
        t_group = group[group["t"] == 1.0]
        c_group = group[group["t"] == 0.0]

        n_t = len(t_group)
        n_c = len(c_group)

        y_t = t_group["y"].sum() if n_t > 0 else 0.0
        y_c = c_group["y"].sum() if n_c > 0 else 0.0

        cr_t = (y_t / n_t) if n_t > 0 else 0.0
        cr_c = (y_c / n_c) if n_c > 0 else 0.0
        observed_lift = cr_t - cr_c
        pred_lift = group["score"].mean()
        inc_conv = observed_lift * n_t

        rows.append({
            "decile": int(d),
            "n_total": n_tot,
            "n_treated": n_t,
            "n_control": n_c,
            "treated_conv_rate": round(cr_t, 4),
            "control_conv_rate": round(cr_c, 4),
            "observed_uplift": round(observed_lift, 4),
            "predicted_uplift_mean": round(pred_lift, 4),
            "incremental_conversions": round(inc_conv, 1),
        })

    return pd.DataFrame(rows)


def policy_value(
    y_true: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    propensity: Union[np.ndarray, pd.Series],
    treatment_policy: Union[np.ndarray, pd.Series],
) -> float:
    """Computes the Inverse Propensity Weighted (IPW) expected policy value.

    V(pi) = (1/N) * sum_{i} [ I(T_i == pi(X_i)) * Y_i / P(T_i == pi(X_i) | X_i) ]

    Args:
        y_true: Binary outcome Y.
        treatment: Observed treatment T (0 or 1).
        propensity: Propensity score e(X) = P(T=1 | X).
        treatment_policy: Prescribed policy decision pi(X) in {0, 1}.

    Returns:
        Expected outcome value under policy pi.
    """
    y = _to_numpy_flat(y_true)
    t = _to_numpy_flat(treatment)
    prop = np.clip(_to_numpy_flat(propensity), 1e-4, 1.0 - 1e-4)
    pi = _to_numpy_flat(treatment_policy)

    # Probability of receiving the treatment that the policy prescribed
    prob_t_matches_pi = np.where(pi == 1.0, prop, 1.0 - prop)
    matches = (t == pi).astype(np.float64)

    weights = matches / prob_t_matches_pi
    return float(np.mean(weights * y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate uplift model on processed dataset.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained PyTorch model .pt")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz processed dataset")
    args = parser.parse_args()

    import torch
    from src.utils import load_checkpoint

    ckpt = load_checkpoint(args.model_path, map_location="cpu")
    data = np.load(args.data)

    X_test = data["X_test"]
    T_test = data["T_test"]
    Y_test = data["Y_test"]

    from src.models import build_model
    model = build_model(
        model_type=ckpt["model_type"],
        input_dim=X_test.shape[1],
        hidden_dim=ckpt["config"]["model"]["hidden_dim"],
        n_layers=ckpt["config"]["model"]["n_layers"],
        dropout=ckpt["config"]["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        uplift_scores = model.predict_uplift(torch.tensor(X_test, dtype=torch.float32)).numpy()

    _, _, _, qini = qini_curve(Y_test, T_test, uplift_scores)
    auuc = auuc_score(Y_test, T_test, uplift_scores)
    deciles = uplift_decile_table(Y_test, T_test, uplift_scores)

    print("\n" + "=" * 50)
    print(f" Uplift Evaluation: {ckpt['model_type'].upper()}")
    print("=" * 50)
    print(f" Qini Coefficient : {qini:.4f}")
    print(f" AUUC Score       : {auuc:.4f}")
    print("\nDecile Summary Table:")
    print(deciles.to_string(index=False))
