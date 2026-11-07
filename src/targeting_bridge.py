"""TargetingBridge: Decoupled Commercial Decision Engine for Promo Allocation.

Translates raw causal CATE predictions into budget-constrained commercial decisions:
- Four-cohort behavioral classification (Persuadable, Sure Thing, Lost Cause, Sleeping Dog)
- Hard campaign budget constraint enforcement
- Sleeping dog protective exclusions
- Expected incremental GMV, margin, and campaign ROI estimation
- Budget vs. Incremental Conversion frontier analysis
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class TargetingBridge:
    """Decoupled commercial layer that operationalizes causal uplift predictions."""

    def __init__(
        self,
        budget: float = 50000.0,
        cost_per_treatment: float = 20.0,
        avg_margin: float = 120.0,
        min_uplift: float = 0.015,
        exclude_sleeping_dogs: bool = True,
        sleeping_dog_threshold: float = -0.01,
    ):
        """Initializes the commercial targeting bridge.

        Args:
            budget: Total allocated campaign budget (in currency units, e.g., INR).
            cost_per_treatment: Marginal cost per coupon or promo sent (e.g., SMS cost + discount face value).
            avg_margin: Gross profit margin realization per incremental order (in currency units).
            min_uplift: Minimum predicted uplift threshold to warrant marketing spend.
            exclude_sleeping_dogs: Guardrail to exclude customers with negative treatment effects.
            sleeping_dog_threshold: Threshold below which a customer is flagged as Sleeping Dog.
        """
        self.budget = float(budget)
        self.cost_per_treatment = float(cost_per_treatment)
        self.avg_margin = float(avg_margin)
        self.min_uplift = float(min_uplift)
        self.exclude_sleeping_dogs = bool(exclude_sleeping_dogs)
        self.sleeping_dog_threshold = float(sleeping_dog_threshold)

    @property
    def max_treatments(self) -> int:
        """Maximum number of customer treatments permitted within budget."""
        if self.cost_per_treatment <= 0:
            return 0
        return int(self.budget // self.cost_per_treatment)

    def classify_cohort(
        self,
        uplift: float,
        y0: Optional[float] = None,
        y1: Optional[float] = None,
    ) -> str:
        """Classifies a single customer into one of four behavioral uplift cohorts."""
        if uplift <= self.sleeping_dog_threshold:
            return "Sleeping Dog"
        elif uplift >= self.min_uplift:
            return "Persuadable"
        elif y0 is not None and y0 >= 0.35:
            return "Sure Thing"
        else:
            return "Lost Cause"

    def classify_cohorts(
        self,
        uplift_scores: np.ndarray,
        y0: Optional[np.ndarray] = None,
        y1: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized classification into Persuadable, Sure Thing, Lost Cause, or Sleeping Dog."""
        n = len(uplift_scores)
        cohorts = np.empty(n, dtype=object)

        if y0 is None:
            # Estimate base conversion proxy if y0 is not passed
            y0 = np.full(n, 0.20)

        for i in range(n):
            cohorts[i] = self.classify_cohort(
                uplift=float(uplift_scores[i]),
                y0=float(y0[i]) if y0 is not None else None,
                y1=float(y1[i]) if y1 is not None else None,
            )
        return cohorts

    def select(
        self,
        uplift_scores: Union[np.ndarray, list, pd.Series],
        customer_ids: Optional[Union[List[Any], np.ndarray, pd.Series]] = None,
        y0: Optional[np.ndarray] = None,
        y1: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Selects optimal customer cohort subject to budget constraints and guardrails.

        Returns DataFrame containing:
            customer_id, uplift_score, cohort, selected, treatment_cost,
            expected_incremental_margin, expected_net_value
        """
        scores = np.asarray(uplift_scores, dtype=np.float64).ravel()
        n = len(scores)

        if customer_ids is None:
            c_ids = [f"CUST_{i+1:06d}" for i in range(n)]
        else:
            c_ids = list(customer_ids)

        cohorts = self.classify_cohorts(scores, y0=y0, y1=y1)

        df = pd.DataFrame({
            "customer_id": c_ids,
            "uplift_score": scores,
            "cohort": cohorts,
        })

        if y0 is not None:
            df["y0_hat"] = np.asarray(y0, dtype=np.float64).ravel()
        if y1 is not None:
            df["y1_hat"] = np.asarray(y1, dtype=np.float64).ravel()

        # Sort candidate ranking primarily by uplift descending
        df = df.sort_values(by="uplift_score", ascending=False).reset_index(drop=True)

        # Eligibility criteria
        eligible_mask = df["uplift_score"] >= self.min_uplift
        if self.exclude_sleeping_dogs:
            eligible_mask = eligible_mask & (df["cohort"] != "Sleeping Dog")

        # Budget capacity constraint (top-K selection)
        k = self.max_treatments
        df["selected"] = False

        eligible_indices = df[eligible_mask].index[:k]
        df.loc[eligible_indices, "selected"] = True

        # Commercial metrics
        df["treatment_cost"] = np.where(df["selected"], self.cost_per_treatment, 0.0)
        df["expected_incremental_margin"] = np.where(
            df["selected"], df["uplift_score"] * self.avg_margin, 0.0
        )
        df["expected_net_value"] = df["expected_incremental_margin"] - df["treatment_cost"]

        return df

    def estimate_roi(self, decisions_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates commercial P&L impact from selected audience."""
        selected = decisions_df[decisions_df["selected"]]
        n_selected = len(selected)

        total_spend = n_selected * self.cost_per_treatment
        expected_conversions = float(selected["uplift_score"].sum()) if n_selected > 0 else 0.0
        expected_margin = expected_conversions * self.avg_margin
        net_profit = expected_margin - total_spend
        roi = (net_profit / total_spend) if total_spend > 0 else 0.0

        cohort_counts = decisions_df["cohort"].value_counts().to_dict()

        return {
            "budget_limit": self.budget,
            "total_audience": len(decisions_df),
            "selected_count": n_selected,
            "actual_spend": round(total_spend, 2),
            "expected_incremental_conversions": round(expected_conversions, 1),
            "expected_incremental_margin": round(expected_margin, 2),
            "net_profit": round(net_profit, 2),
            "roi": round(roi, 4),
            "roi_percent": round(roi * 100.0, 2),
            "cohort_breakdown": cohort_counts,
        }

    def budget_frontier(
        self,
        uplift_scores: np.ndarray,
        steps: int = 10,
        max_budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Simulates campaign ROI and incremental orders across varying budget levels.

        Generates the diminishing-returns frontier curve for executive decision-making.
        """
        ceiling = max_budget if max_budget is not None else (self.budget * 2.0)
        budgets = np.linspace(ceiling / steps, ceiling, steps)

        rows = []
        for b in budgets:
            sim_bridge = TargetingBridge(
                budget=b,
                cost_per_treatment=self.cost_per_treatment,
                avg_margin=self.avg_margin,
                min_uplift=self.min_uplift,
                exclude_sleeping_dogs=self.exclude_sleeping_dogs,
                sleeping_dog_threshold=self.sleeping_dog_threshold,
            )
            sim_decisions = sim_bridge.select(uplift_scores)
            pnl = sim_bridge.estimate_roi(sim_decisions)
            rows.append({
                "budget": round(b, 0),
                "targeted_users": pnl["selected_count"],
                "expected_incremental_orders": pnl["expected_incremental_conversions"],
                "net_profit": pnl["net_profit"],
                "roi_percent": pnl["roi_percent"],
            })

        return pd.DataFrame(rows)

