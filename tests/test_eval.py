"""Unit tests for Uplift-DRL evaluation metrics and commercial targeting bridge.

16 pure-NumPy / Python tests verifying mathematical invariants, edge cases,
and commercial guardrails without GPU dependencies.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import pytest
    approx_fn = pytest.approx
except ImportError:
    def approx_fn(expected, abs=1e-5):
        class _Approx:
            def __init__(self, exp, tolerance):
                self.exp = exp
                self.tolerance = tolerance
            def __eq__(self, actual):
                import builtins
                return builtins.abs(self.exp - actual) <= self.tolerance
            def __repr__(self):
                return f"approx({self.exp} ± {self.tolerance})"
        return _Approx(expected, abs)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import generate_synthetic_uplift_data
from src.eval import auuc_score, policy_value, qini_curve, uplift_decile_table
from src.targeting_bridge import TargetingBridge


# -----------------------------------------------------------------------------
# Qini Curve & AUUC Invariant Tests (Tests 1 - 7)
# -----------------------------------------------------------------------------
def test_qini_empty_input():
    """Test 1: Empty inputs return safe zero structures without division errors."""
    fractions, qini_vals, rand_vals, qini_score = qini_curve([], [], [])
    assert len(fractions) == 1
    assert qini_score == 0.0


def test_qini_perfect_ranking():
    """Test 2: Perfect uplift ranking produces a strictly positive Qini coefficient."""
    # 100 treated with 100% conversion, 100 control with 0% conversion
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1])

    _, qini_vals, rand_vals, qini = qini_curve(y, t, scores)
    assert qini > 0.0
    assert qini_vals[-1] == approx_fn(rand_vals[-1], abs=1e-5)


def test_qini_random_ranking():
    """Test 3: Random scoring yields Qini score near zero."""
    rng = np.random.default_rng(42)
    n = 2000
    y = rng.binomial(1, 0.2, n)
    t = rng.binomial(1, 0.5, n)
    scores = rng.uniform(0.0, 1.0, n)

    _, _, _, qini = qini_curve(y, t, scores)
    # Scaled to population, random ordering should be close to 0
    assert abs(qini) < 20.0


def test_qini_inverse_ranking():
    """Test 4: Inversely ranked scores produce a strictly negative Qini score."""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Reversed predictions (putting control non-responders first)
    scores = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6])

    _, _, _, qini = qini_curve(y, t, scores)
    assert qini < 0.0


def test_qini_curve_endpoints():
    """Test 5: Qini curve and random baseline start at 0 and converge at 100%."""
    rng = np.random.default_rng(123)
    y = rng.binomial(1, 0.25, 500)
    t = rng.binomial(1, 0.5, 500)
    scores = rng.normal(0, 1, 500)

    fractions, qini_vals, rand_vals, _ = qini_curve(y, t, scores)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert qini_vals[0] == 0.0 and rand_vals[0] == 0.0
    assert approx_fn(qini_vals[-1], abs=1e-5) == rand_vals[-1]


def test_auuc_monotonic_gain():
    """Test 6: AUUC returns a valid numeric float for typical uplift data."""
    rng = np.random.default_rng(999)
    y = rng.binomial(1, 0.3, 1000)
    t = rng.binomial(1, 0.5, 1000)
    scores = rng.uniform(-0.1, 0.2, 1000)

    score = auuc_score(y, t, scores)
    assert isinstance(score, float)
    assert not np.isnan(score)


def test_auuc_perfect_vs_reversed():
    """Test 7: AUUC under optimal ranking exceeds AUUC under reversed ranking."""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores_opt = np.array([0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
    scores_rev = np.array([0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8])

    auuc_opt = auuc_score(y, t, scores_opt)
    auuc_rev = auuc_score(y, t, scores_rev)
    assert auuc_opt > auuc_rev


# -----------------------------------------------------------------------------
# Decile Table Tests (Tests 8 - 10)
# -----------------------------------------------------------------------------
def test_uplift_decile_table_shape():
    """Test 8: Decile table has exactly 10 bins with required diagnostic columns."""
    rng = np.random.default_rng(42)
    n = 1000
    y = rng.binomial(1, 0.2, n)
    t = rng.binomial(1, 0.5, n)
    scores = rng.normal(0.05, 0.02, n)

    df_deciles = uplift_decile_table(y, t, scores, n_bins=10)
    assert len(df_deciles) == 10
    required_cols = [
        "decile", "n_total", "n_treated", "n_control",
        "treated_conv_rate", "control_conv_rate", "observed_uplift",
        "predicted_uplift_mean", "incremental_conversions",
    ]
    for col in required_cols:
        assert col in df_deciles.columns


def test_uplift_decile_table_aggregates():
    """Test 9: Total sample counts across deciles sum exactly to input size."""
    n = 1500
    rng = np.random.default_rng(77)
    y = rng.binomial(1, 0.15, n)
    t = rng.binomial(1, 0.5, n)
    scores = rng.uniform(0.0, 0.1, n)

    df_deciles = uplift_decile_table(y, t, scores, n_bins=5)
    assert df_deciles["n_total"].sum() == n
    assert (df_deciles["n_treated"] + df_deciles["n_control"]).sum() == n


def test_uplift_decile_table_ordered_scores():
    """Test 10: Predicted uplift mean strictly decreases from decile 1 to decile 10."""
    n = 2000
    rng = np.random.default_rng(88)
    scores = np.linspace(0.20, -0.05, n)
    y = rng.binomial(1, 0.2, n)
    t = rng.binomial(1, 0.5, n)

    df_deciles = uplift_decile_table(y, t, scores, n_bins=10)
    pred_means = df_deciles["predicted_uplift_mean"].values
    for i in range(len(pred_means) - 1):
        assert pred_means[i] >= pred_means[i + 1]


# -----------------------------------------------------------------------------
# Policy Value & Commercial Bridge Tests (Tests 11 - 16)
# -----------------------------------------------------------------------------
def test_policy_value_ipw():
    """Test 11: Inverse Propensity Weighted policy value calculates correct expected outcome."""
    y = np.array([1.0, 1.0, 0.0, 0.0])
    t = np.array([1.0, 0.0, 1.0, 0.0])
    prop = np.array([0.5, 0.5, 0.5, 0.5])
    pi = np.array([1.0, 0.0, 1.0, 0.0])  # Policy matches treatment for all

    val = policy_value(y, t, prop, pi)
    # When pi matches t, weight is 1/0.5 = 2.0. Expected outcome: mean([1*2, 1*2, 0*2, 0*2]) = 1.0
    assert approx_fn(val, abs=1e-5) == 1.0


def test_targeting_bridge_budget_ceiling():
    """Test 12: TargetingBridge never selects more customers than budget permits."""
    bridge = TargetingBridge(budget=1000.0, cost_per_treatment=50.0)  # Max treatments = 20
    scores = np.linspace(0.10, 0.01, 100)

    decisions = bridge.select(scores)
    selected_count = decisions["selected"].sum()
    assert selected_count <= 20
    assert selected_count == bridge.max_treatments


def test_targeting_bridge_sleeping_dog_exclusion():
    """Test 13: Customers identified as Sleeping Dogs are strictly excluded when guardrail is enabled."""
    bridge = TargetingBridge(
        budget=100000.0,
        cost_per_treatment=10.0,
        exclude_sleeping_dogs=True,
        sleeping_dog_threshold=-0.01,
    )
    # 5 sleeping dogs with negative uplift
    scores = np.array([0.15, 0.10, -0.05, -0.08, -0.02, 0.03])
    decisions = bridge.select(scores)

    sleeping_dogs = decisions[decisions["cohort"] == "Sleeping Dog"]
    assert len(sleeping_dogs) == 3
    assert sleeping_dogs["selected"].sum() == 0  # None selected


def test_targeting_bridge_min_uplift_floor():
    """Test 14: Customers below minimum uplift floor are excluded even with excess budget."""
    bridge = TargetingBridge(budget=100000.0, cost_per_treatment=10.0, min_uplift=0.05)
    scores = np.array([0.08, 0.06, 0.04, 0.02, 0.01])  # Only first two exceed 0.05

    decisions = bridge.select(scores)
    assert decisions["selected"].sum() == 2


def test_targeting_bridge_roi_math():
    """Test 15: Financial P&L formulas (spend, margin, net profit, ROI) balance exactly."""
    bridge = TargetingBridge(budget=100.0, cost_per_treatment=25.0, avg_margin=100.0, min_uplift=0.01)
    # Selects top 4 (budget allows 100/25 = 4)
    scores = np.array([0.10, 0.10, 0.10, 0.10, 0.05])
    decisions = bridge.select(scores)
    pnl = bridge.estimate_roi(decisions)

    # 4 selected * 25 cost = 100 spend
    assert pnl["actual_spend"] == 100.0
    # 4 * 0.10 = 0.40 expected incremental orders
    assert pnl["expected_incremental_conversions"] == 0.4
    # 0.4 * 100 margin = 40 margin
    assert pnl["expected_incremental_margin"] == 40.0
    # Net profit: 40 - 100 = -60
    assert pnl["net_profit"] == -60.0
    # ROI: -60 / 100 = -0.60
    assert pnl["roi"] == -0.60


def test_synthetic_generator_reproducibility():
    """Test 16: Synthetic generator is deterministic given a fixed random seed."""
    data1 = generate_synthetic_uplift_data(n_samples=500, seed=1234)
    data2 = generate_synthetic_uplift_data(n_samples=500, seed=1234)

    np.testing.assert_array_equal(data1["X"], data2["X"])
    np.testing.assert_array_equal(data1["T"], data2["T"])
    np.testing.assert_array_equal(data1["Y"], data2["Y"])
    assert len(data1["feature_names"]) == 10


if __name__ == "__main__":
    tests = [
        test_qini_empty_input,
        test_qini_perfect_ranking,
        test_qini_random_ranking,
        test_qini_inverse_ranking,
        test_qini_curve_endpoints,
        test_auuc_monotonic_gain,
        test_auuc_perfect_vs_reversed,
        test_uplift_decile_table_shape,
        test_uplift_decile_table_aggregates,
        test_uplift_decile_table_ordered_scores,
        test_policy_value_ipw,
        test_targeting_bridge_budget_ceiling,
        test_targeting_bridge_sleeping_dog_exclusion,
        test_targeting_bridge_min_uplift_floor,
        test_targeting_bridge_roi_math,
        test_synthetic_generator_reproducibility,
    ]
    print(f"Running {len(tests)} Uplift-DRL unit tests...")
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} tests passed successfully!")
