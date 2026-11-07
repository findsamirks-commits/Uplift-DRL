"""Loss functions and targeted regularization for CATE / Uplift models.

Implements the Targeted Regularization term from:
Shi, Blei & Veitch (2019) — 'Adapting Neural Networks for the Estimation of Treatment Effects' (NeurIPS).

The targeted regularization term couples the outcome heads with the propensity score
via a scalar epsilon and a clever covariate H(t, x), guaranteeing the non-parametric
efficiency bound and providing double-robust estimation of the average treatment effect.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DragonNetLoss(nn.Module):
    """Targeted Regularization loss for DragonNet.

    L_total = L_outcome(Y, y_t) + alpha * L_propensity(T, g) + beta * L_targeted(Y, y_tilde)

    where:
        L_outcome: Binary cross-entropy on observed outcome for the received treatment arm.
        L_propensity: Binary cross-entropy on treatment assignment.
        L_targeted: Mean squared error on the perturbed prediction y_tilde = y_obs + epsilon * H(t, x).
        H(t, x) = (t / g(x)) - ((1 - t) / (1 - g(x))) is the clever covariate from TMLE.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps_clip: float = 1e-3):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps_clip = float(eps_clip)

    def forward(
        self,
        y_true: torch.Tensor,
        treatment: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        propensity: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculates combined DragonNet loss.

        Args:
            y_true: Observed ground-truth outcome [N, 1].
            treatment: Observed binary treatment [N, 1].
            y0_pred: Predicted control outcome head P(Y=1 | X, T=0) [N, 1].
            y1_pred: Predicted treated outcome head P(Y=1 | X, T=1) [N, 1].
            propensity: Predicted propensity score P(T=1 | X) [N, 1].
            epsilon: Learned scalar parameter [1].

        Returns:
            total_loss: Scalar loss tensor for backpropagation.
            metrics: Dictionary of decomposed loss components for logging.
        """
        # 1. Outcome loss on observed treatment arms
        loss_y0 = F.binary_cross_entropy(y0_pred, y_true, reduction="none")
        loss_y1 = F.binary_cross_entropy(y1_pred, y_true, reduction="none")
        # Mask by observed treatment: (1 - T) * L0 + T * L1
        outcome_loss = torch.mean((1.0 - treatment) * loss_y0 + treatment * loss_y1)

        # 2. Propensity loss (Binary cross-entropy on treatment assignment)
        prop_clipped = torch.clamp(propensity, self.eps_clip, 1.0 - self.eps_clip)
        propensity_loss = F.binary_cross_entropy(prop_clipped, treatment)

        # 3. Targeted Regularization via clever covariate H(t, x)
        # H(t, x) = (t / g(x)) - ((1 - t) / (1 - g(x)))
        clever_covariate = (treatment / prop_clipped) - ((1.0 - treatment) / (1.0 - prop_clipped))

        # Factual outcome prediction
        y_factual = (1.0 - treatment) * y0_pred + treatment * y1_pred

        # Perturbed outcome estimate
        y_tilde = y_factual + epsilon * clever_covariate

        # Targeted regularization loss (enforces solving the efficient score equation)
        targeted_loss = torch.mean((y_true - y_tilde) ** 2)

        # Total combined loss
        total_loss = outcome_loss + (self.alpha * propensity_loss) + (self.beta * targeted_loss)

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_outcome": float(outcome_loss.item()),
            "loss_propensity": float(propensity_loss.item()),
            "loss_targeted": float(targeted_loss.item()),
            "epsilon": float(epsilon.item()),
        }
        return total_loss, metrics


class TARNetLoss(nn.Module):
    """Loss for Treatment-Agnostic Representation Network (TARNet)."""

    def forward(
        self,
        y_true: torch.Tensor,
        treatment: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_y0 = F.binary_cross_entropy(y0_pred, y_true, reduction="none")
        loss_y1 = F.binary_cross_entropy(y1_pred, y_true, reduction="none")
        total_loss = torch.mean((1.0 - treatment) * loss_y0 + treatment * loss_y1)
        metrics = {"loss_total": float(total_loss.item())}
        return total_loss, metrics


class StandardUpliftLoss(nn.Module):
    """Binary cross entropy loss for S-Learner or T-Learner factual predictions."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = F.binary_cross_entropy(y_pred, y_true)
        return loss, {"loss_total": float(loss.item())}

