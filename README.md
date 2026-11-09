# Uplift-DRL 🎯🧠
**Deep Causal Learning & CATE Estimation for Retail Promo Targeting — powered by DragonNet, TARNet & Targeted Regularization.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uplift-drl.streamlit.app)
[![Live Demo](https://img.shields.io/badge/demo-live-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://uplift-drl.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-findsamirks--commits-black?style=flat&logo=github)](https://github.com/findsamirks-commits/Uplift-DRL)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Causal CATE](https://img.shields.io/badge/Causal_CATE-DragonNet-9146FF?style=flat&logo=pytorch&logoColor=white)](https://github.com/findsamirks-commits/Uplift-DRL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🌐 **Live Interactive Streamlit Demo:** [uplift-drl.streamlit.app](https://uplift-drl.streamlit.app)

---

## 📖 About Uplift-DRL

**Uplift-DRL** is an enterprise-grade deep causal machine learning pipeline designed to answer the single most expensive question in quick-commerce and e-commerce growth merchandising:

> *"Which customers will convert **only if** given a promo — and which would have converted anyway?"*

Rather than predicting standard response propensity \(P(Y=1 \mid X)\) — which routinely wastes marketing budgets on customers who need no discount — Uplift-DRL estimates the **Conditional Average Treatment Effect (CATE)** at individual customer resolution:

\[
\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X=x]
\]

### 🏛️ Core Architectural Pillars

- **Four Unified CATE Estimators:** Modular implementations of **DragonNet** (Shi et al. 2019), **TARNet** (Shalit et al. 2017), **T-Learner**, and **S-Learner** sharing a single `UpliftModel` base class for zero-friction benchmarking.
- **Double-Robust Targeted Regularization:** Implements the true Targeted Maximum Likelihood Estimation (TMLE) scalar \(\varepsilon\) and clever covariate \(H(t, x)\) to solve the efficient score equation and guarantee double-robust CATE estimation on observational retail data.
- **Decoupled Commercial TargetingBridge:** Bridges causal ML and commercial P&L by enforcing hard campaign budget ceilings, estimating expected incremental gross margins, and strictly excluding "Sleeping Dogs" (customers where promos trigger unsubscribes or notification fatigue).
- **Pure-NumPy Diagnostic Engine:** High-performance implementations of Qini Curves, Qini Coefficient, AUUC, Uplift Decile Tables, and Inverse Propensity Weighted (IPW) Expected Policy Value without external causal dependencies.
- **Autonomous Retail AI Mesh Ready:** Ingests markdown elasticities from `Multi-Agent-Dynamic-Pricing-Swarm` and pushes high-uplift Persuadable shopper cohorts to `BasketGNN` for targeted cross-sell boosts.

---

## 🎯 Why uplift, not just classification?

Most promo targeting models predict **P(conversion | treatment)**. This is wrong — it targets the *obvious* buyers who would have converted anyway, wasting promo budget on **"sure things"**.

Uplift modeling estimates the **Individual Treatment Effect (ITE)**:

\[
\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X=x]
\]

…and sorts customers into **four cohorts**:

| Cohort | Treated outcome | Control outcome | Action |
|---|---|---|---|
| 🟢 **Persuadable** | Buys | Doesn't buy | **Target** — pure incremental ROI |
| ⚪ **Sure Thing** | Buys | Buys | **Skip** — wasted spend |
| ⚫ **Lost Cause** | Doesn't buy | Doesn't buy | **Skip** — no response |
| 🔴 **Sleeping Dog** | Doesn't buy | Buys | **Never target** — promo *backfires* |

The whole point: **spend the next ₹1 of coupon budget on the persuadable with the highest uplift**, not on the customer with the highest base conversion rate.

---

## 🏗️ Four models, one interface

All four are implemented in PyTorch with a common `UpliftModel` base class:

| Model | Architecture | Loss | Reference |
|---|---|---|---|
| **S-Learner** | One MLP on `[X, T]` → `Y` | `BCE(Y, Ŷ)` | Künzel et al. 2019 |
| **T-Learner** | Two MLPs: one on treated, one on control | `BCE_0 + BCE_1` | Künzel et al. 2019 |
| **TARNet** | Shared φ(X) + two outcome heads h₀, h₁ | `BCE(Y, h_T(φ(X)))` | Shalit et al. 2017 |
| **DragonNet** | TARNet + propensity head g(φ(X)) + **targeted regularization** | TARNet + π-loss + ε-term | Shi, Blei & Veitch 2019 |

**DragonNet** is the recommended default — its propensity head acts as a regularizer that keeps φ(X) *predictive of treatment*, which is exactly what you need for unbiased CATE on observational data.

---

## 📁 Repository layout

```
Uplift-DRL/
├── configs/
│   └── uplift.yaml                 # Hyperparameters per model
├── data/
│   └── README.md                   # Criteo-Uplift v2 / Hillstrom / Starbucks instructions
├── notebooks/
│   └── 01_data_exploration.ipynb   # Treatment balance + outcome distributions
├── src/
│   ├── __init__.py
│   ├── data.py                     # Loaders for Criteo / Hillstrom / custom CSV / synthetic
│   ├── models.py                   # S-Learner, T-Learner, TARNet, DragonNet
│   ├── losses.py                   # Targeted regularization for DragonNet
│   ├── train.py                    # Unified CLI: --model {s|t|tarnet|dragonnet}
│   ├── eval.py                     # Qini, AUUC, decile table, Policy Value
│   ├── inference.py                # UpliftPredictor for scoring new customers
│   ├── targeting_bridge.py         # Budget-constrained top-K + ROI estimator
│   └── utils.py
├── app/
│   └── streamlit_app.py            # Interactive Qini curve + persuadable scoring
├── tests/
│   └── test_eval.py                # Qini + AUUC correctness tests
└── docs/
    └── architecture.md             # Model math + design decisions
```

---

## 🚀 Quickstart

```bash
# 1. Install
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                      # macOS/Linux
pip install -r requirements.txt

# 2. Download Hillstrom (smallest) — see data/README.md
# Place at data/hillstrom.csv (or use --source synthetic to generate data immediately)

# 3. Prep → train DragonNet → eval → score
python -m src.data --source synthetic --output data/processed/synthetic.npz
python -m src.train --model dragonnet --data data/processed/synthetic.npz --config configs/uplift.yaml
python -m src.eval  --model-path checkpoints/dragonnet_best.pt --data data/processed/synthetic.npz
python -m src.inference --model-path checkpoints/dragonnet_best.pt --input data/scoring_sample.csv --output uplift_scores.csv

# 4. Interactive demo
streamlit run app/streamlit_app.py
```

---

## 📊 Evaluation metrics

Uplift evaluation is *not* standard classification AUC — true labels for counterfactual individual treatment effects are unobservable at the individual level (the Fundamental Problem of Causal Inference). Instead:

| Metric | What it measures | Formula / Implementation |
|---|---|---|
| **Qini curve** | Cumulative incremental gain vs % of population targeted | \(Q(k) = Y_{T, k} - Y_{C, k} \cdot \frac{N_{T, k}}{N_{C, k}}\) |
| **Qini coefficient (Q)** | Area between Qini curve and random-targeting baseline | Area under model curve minus random baseline area |
| **AUUC** (Area Under Uplift Curve) | Cumulative uplift gain metric, less sensitive to treatment imbalance | Area under \(U(k) = (P_T(k) - P_C(k)) \cdot N(k)\) |
| **Uplift decile table** | Gain in treatment-control conversion difference across score deciles | Binned conversion rate delta (\(Y_T/N_T - Y_C/N_C\)) |
| **Expected Policy Value** | Population value under greedy policy \(\pi(X) = \mathbb{I}(\hat{\tau}(X) > \tau_{\text{thresh}})\) | Inverse Propensity Weighted (IPW / Horvitz-Thompson) |

All implemented in pure NumPy in `src/eval.py` — zero heavy external dependencies.

---

## 🧠 Targeting bridge (business layer)

`src/targeting_bridge.py` decouples the learned CATE from the *actual* commercial targeting decision:

```python
from src.targeting_bridge import TargetingBridge

bridge = TargetingBridge(
    budget=50_000,              # ₹ promo budget
    cost_per_treatment=20,      # ₹ per coupon fulfilled
    avg_margin=120,             # ₹ margin per incremental conversion
    min_uplift=0.02,            # Floor: skip anyone below 2 pp lift
    exclude_sleeping_dogs=True
)

decisions = bridge.select(uplift_scores, customer_ids)
# → DataFrame with [customer_id, uplift, selected, cohort, expected_incremental_value]
roi = bridge.estimate_roi(decisions)
# → {'spend': 50000, 'expected_conversions': 412, 'expected_margin': 49440, 'net_profit': -560, 'roi': -0.011}
```

This separation means:
- **Data scientists** iterate on CATE model quality (Qini, AUUC, calibration)
- **Category managers** iterate on budget, margin assumptions, sleeping-dog guardrails
- **A/B testing** can hold the model fixed while varying budget bands and coupon values

---

## 🔗 Fits into the Autonomous Retail AI Mesh

```
┌─────────────────────────────────────────────────────────┐
│          Multi-Agent Dynamic Pricing Swarm              │
│   (optimizes WHAT price / discount depth to offer)      │
└────────────────────────┬────────────────────────────────┘
                         │ promo_depth, SKU, category margin
                         ▼
┌─────────────────────────────────────────────────────────┐
│     Uplift-DRL  (DragonNet Deep Causal CATE)            │
│   (decides WHO to send the promo to, under budget)      │
└──────┬──────────────────────────────────┬───────────────┘
       │ persuadable_user_ids             │ campaign ROI forecast
       ▼                                  ▼
┌──────────────────┐             ┌──────────────────────┐
│ BasketGNN        │             │ Executive Dashboard  │
│ (boosts recs for │             │ (budget vs uplift    │
│  persuadables)   │             │  frontier curve)     │
└──────────────────┘             └──────────────────────┘
```

**Causal inference + price optimization + GNN recommendations = a closed-loop retail intelligence stack.**

---

## 📚 Datasets

Full instructions in `data/README.md`. Supported out of the box:

| Dataset | Rows | Treatment | Outcome | Notes |
|---|---|---|---|---|
| [Hillstrom MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | 64K | 3-arm (men/women/no-email) | Visit, conversion, spend | Best benchmark starter |
| [Criteo-Uplift v2](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | 13.9M | Binary ad impression | Visit, conversion | Industry gold standard |
| [Starbucks Promo](https://www.kaggle.com/datasets/snowman6210/starbucks) | 84K | Binary mobile coupon | Offer response & spending | Quick-commerce flavor |
| **Synthetic Quick-Commerce** | Configurable | Binary targeted voucher | Next-day reorder conversion | Generates immediately without downloads |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

16 tests covering Qini area calculation, AUUC monotonicity invariants, decile aggregation, and targeting-bridge budget enforcement — all pure Python, no GPU required.

---

## 🛠️ Config

`configs/uplift.yaml`:

```yaml
model:
  hidden_dim: 128
  dropout: 0.2
  n_layers: 3

dragonnet:
  alpha: 1.0               # Weight on treatment prediction loss (propensity cross-entropy)
  beta: 1.0                # Weight on targeted regularization (clever covariate term)

train:
  lr: 1e-3
  weight_decay: 1e-4
  batch_size: 512
  epochs: 50
  early_stopping_patience: 5
  val_metric: qini

targeting:
  default_budget: 50000
  default_cost_per_treatment: 20
  default_margin: 120
  min_uplift: 0.015
  exclude_sleeping_dogs: true
```

---

## 📖 References

- Künzel, Sekhon, Bickel, Yu (2019) — *Metalearners for estimating heterogeneous treatment effects using machine learning*. PNAS.
- Shalit, Johansson, Sontag (2017) — *Estimating individual treatment effect: generalization bounds and algorithms (TARNet)*. ICML.
- Shi, Blei, Veitch (2019) — *Adapting Neural Networks for the Estimation of Treatment Effects (DragonNet)*. NeurIPS.
- Diemert, Betlei, Renaudin, Amini (2018) — *A Large Scale Benchmark for Uplift Modeling*. KDD AdKDD & TargetAd.
- Radcliffe & Surry (2011) — *Real-world uplift modelling with significance-based uplift trees*. White Paper.

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

**Built by [Samir Sandhwar](https://github.com/findsamirks-commits)** — Category Head at Milkbasket, IIT Madras Agentic AI alumnus. Part of the [Autonomous Retail AI Mesh](https://github.com/findsamirks-commits).

