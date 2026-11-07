"""PyTorch Deep Causal Learning Models for CATE and Uplift Modeling.

Implements four estimators sharing a unified UpliftModel base class:
1. S-Learner: Single MLP on concatenated [X, T]
2. T-Learner: Two independent MLPs for Control and Treated cohorts
3. TARNet: Shared representation phi(X) with bifurcated outcome heads (Shalit et al. 2017)
4. DragonNet: TARNet + Propensity head + Targeted Regularization epsilon (Shi, Blei & Veitch 2019)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpliftModel(nn.Module, ABC):
    """Abstract base class for all Uplift/CATE estimators."""

    @abstractmethod
    def predict_outcomes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts counterfactual outcomes (y0_hat, y1_hat) for feature tensor x."""
        pass

    def predict_uplift(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts Individual Treatment Effect / CATE: tau_hat(x) = y1_hat - y0_hat."""
        y0_hat, y1_hat = self.predict_outcomes(x)
        return y1_hat - y0_hat


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """Helper to build standard MLP with LayerNorm, ReLU, and Dropout."""
    layers = []
    curr_dim = in_dim
    for _ in range(n_layers):
        layers.extend([
            nn.Linear(curr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        curr_dim = hidden_dim
    layers.append(nn.Linear(curr_dim, out_dim))
    return nn.Sequential(*layers)


class SLearner(UpliftModel):
    """S-Learner (Single Learner).

    Treats the treatment assignment T as an ordinary feature concatenated with X.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = _build_mlp(
            in_dim=input_dim + 1,  # [X, T]
            hidden_dim=hidden_dim,
            out_dim=1,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting Y given observed X and T."""
        xt = torch.cat([x, t], dim=-1)
        return torch.sigmoid(self.net(xt))

    def predict_outcomes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts counterfactuals by clamping T=0 and T=1."""
        batch_size = x.shape[0]
        device = x.device
        t0 = torch.zeros((batch_size, 1), device=device)
        t1 = torch.ones((batch_size, 1), device=device)

        y0 = self.forward(x, t0)
        y1 = self.forward(x, t1)
        return y0, y1


class TLearner(UpliftModel):
    """T-Learner (Two Learners).

    Fits two entirely separate neural networks:
    net_0 for control units (T=0) and net_1 for treated units (T=1).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net_0 = _build_mlp(in_dim=input_dim, hidden_dim=hidden_dim, out_dim=1, n_layers=n_layers, dropout=dropout)
        self.net_1 = _build_mlp(in_dim=input_dim, hidden_dim=hidden_dim, out_dim=1, n_layers=n_layers, dropout=dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting observed outcome depending on T."""
        y0 = torch.sigmoid(self.net_0(x))
        y1 = torch.sigmoid(self.net_1(x))
        return torch.where(t > 0.5, y1, y0)

    def predict_outcomes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y0 = torch.sigmoid(self.net_0(x))
        y1 = torch.sigmoid(self.net_1(x))
        return y0, y1


class TARNet(UpliftModel):
    """Treatment-Agnostic Representation Network (Shalit et al. 2017).

    Learns a shared covariate representation phi(X) followed by two separate
    hypothesis heads h0 and h1 for control and treated potential outcomes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        outcome_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Shared representation network phi(X)
        self.representation = _build_mlp(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        # Outcome heads
        self.head_0 = nn.Sequential(
            nn.Linear(hidden_dim, outcome_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outcome_dim, 1),
        )
        self.head_1 = nn.Sequential(
            nn.Linear(hidden_dim, outcome_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outcome_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        phi = self.representation(x)
        y0 = torch.sigmoid(self.head_0(phi))
        y1 = torch.sigmoid(self.head_1(phi))
        if t is not None:
            return torch.where(t > 0.5, y1, y0)
        return y0, y1

    def predict_outcomes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.representation(x)
        y0 = torch.sigmoid(self.head_0(phi))
        y1 = torch.sigmoid(self.head_1(phi))
        return y0, y1


class DragonNet(UpliftModel):
    """DragonNet with Targeted Regularization (Shi, Blei & Veitch 2019).

    Couples a shared representation phi(X) with:
    1. Propensity head g(phi(X)) predicting treatment probability P(T=1 | X)
    2. Bifurcated outcome heads h0(phi(X)) and h1(phi(X))
    3. Targeted regularization scalar parameter epsilon with clever covariate
       enforcing double-robustness and the score equation for average treatment effects.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        outcome_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Shared representation phi(X)
        self.representation = _build_mlp(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Propensity head: g(phi) -> pi_hat in (0, 1)
        self.propensity_head = nn.Sequential(
            nn.Linear(hidden_dim, outcome_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outcome_dim, 1),
        )

        # Outcome heads: h0, h1
        self.head_0 = nn.Sequential(
            nn.Linear(hidden_dim, outcome_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outcome_dim, 1),
        )
        self.head_1 = nn.Sequential(
            nn.Linear(hidden_dim, outcome_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outcome_dim, 1),
        )

        # Targeted regularization scalar parameter epsilon
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (y0_prob, y1_prob, propensity_prob, epsilon)."""
        phi = self.representation(x)
        propensity = torch.sigmoid(self.propensity_head(phi))
        y0 = torch.sigmoid(self.head_0(phi))
        y1 = torch.sigmoid(self.head_1(phi))
        return y0, y1, propensity, self.epsilon

    def predict_propensity(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts propensity score P(T=1 | X)."""
        phi = self.representation(x)
        return torch.sigmoid(self.propensity_head(phi))

    def predict_outcomes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.representation(x)
        y0 = torch.sigmoid(self.head_0(phi))
        y1 = torch.sigmoid(self.head_1(phi))
        return y0, y1


def build_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 128,
    n_layers: int = 3,
    dropout: float = 0.2,
    outcome_dim: int = 64,
) -> UpliftModel:
    """Model factory returning configured UpliftModel instance."""
    m_type = model_type.lower().strip()
    if m_type in ["s", "slearner", "s-learner"]:
        return SLearner(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    elif m_type in ["t", "tlearner", "t-learner"]:
        return TLearner(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    elif m_type in ["tarnet", "tar-net"]:
        return TARNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            outcome_dim=outcome_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
    elif m_type in ["dragonnet", "dragon-net", "dragon"]:
        return DragonNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            outcome_dim=outcome_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from ['s', 't', 'tarnet', 'dragonnet'].")
