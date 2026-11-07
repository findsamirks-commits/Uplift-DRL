# Uplift-DRL Architectural Specification & Mathematical Foundations

This document details the causal inference mathematical foundations, neural architectures, targeted regularization theory, and enterprise integration patterns powering **Uplift-DRL**.

---

## 1. Causal Problem Formulation

In conventional retail classification, supervised models optimize for conditional association:

\[
\hat{P}(Y=1 \mid X=x, T=1)
\]

This formulation exhibits fundamental commercial failure: customers with high baseline order frequency ("Sure Things") receive high scores and absorb coupon budgets, despite zero incremental lift.

### 1.1 Potential Outcomes Framework (Rubin-Neyman Causal Model)

Let each unit (customer) \(i\) possess two potential outcomes:
- \(Y_i^{(1)}\): The potential outcome if the customer receives the marketing promotion (\(T_i = 1\)).
- \(Y_i^{(0)}\): The potential outcome if the customer is held in the control group (\(T_i = 0\)).

The **Individual Treatment Effect (ITE)** is defined as:

\[
\tau_i = Y_i^{(1)} - Y_i^{(0)}
\]

Because only one potential outcome is observed in reality for any customer (\(Y_i = T_i Y_i^{(1)} + (1 - T_i) Y_i^{(0)}\)), the individual effect \(\tau_i\) is fundamentally unobservable (The Fundamental Problem of Causal Inference).

Instead, we estimate the **Conditional Average Treatment Effect (CATE)** given customer covariates \(X \in \mathbb{R}^d\):

\[
\tau(x) = \mathbb{E}\left[ Y^{(1)} - Y^{(0)} \;\middle|\; X = x \right]
\]

### 1.2 Identification Assumptions

CATE is non-parametrically identifiable from observational and experimental data under standard causal assumptions:

1. **Unconfoundedness (Conditional Ignorability):**
   \[
   (Y^{(0)}, Y^{(1)}) \perp T \mid X
   \]
   All confounding covariates affecting both treatment assignment and outcome are recorded in \(X\).

2. **Overlap (Positivity):**
   \[
   0 < e(x) < 1 \quad \forall x \in \text{supp}(X), \quad \text{where } e(x) \triangleq P(T=1 \mid X=x)
   \]
   Every customer profile has a non-zero probability of receiving treatment or control.

3. **Stable Unit Treatment Value Assumption (SUTVA):**
   No interference between units (one shopper's coupon does not alter another shopper's purchase propensity), and no hidden treatment variations.

---

## 2. Four Estimator Architectures

All estimators implement a shared `UpliftModel` base class to allow drop-in comparative experimentation.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 UpliftModel Base Class                 │
                  │  predict_outcomes(x) -> (y0, y1)                       │
                  │  predict_uplift(x)   -> tau = y1 - y0                  │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
         ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
         ▼                   ▼                               ▼                   ▼
    S-Learner           T-Learner                         TARNet              DragonNet
[X, T] -> MLP -> Y   Two MLPs: f0(X), f1(X)         Shared phi(X)       Shared phi(X)
(Künzel et al. 2019) (Künzel et al. 2019)            Outcome heads       + Propensity head g(phi)
                                                    (Shalit et al. 2017) + Targeted Regularization
                                                                        (Shi, Blei & Veitch 2019)
```

### 2.1 S-Learner (Single Learner)
A single model \(\mu(X, T)\) trained on concatenated features and treatment:
\[
\hat{\tau}_{\text{S}}(x) = \mu(x, 1) - \mu(x, 0)
\]
*Limitation:* When \(X\) is high-dimensional and \(T\) is 1-dimensional, standard neural networks often regularize the treatment signal towards zero, underestimating uplift.

### 2.2 T-Learner (Two Learners)
Two entirely separate networks trained on respective treatment sub-populations:
\[
\mu_0(x) = \mathbb{E}[Y \mid X=x, T=0], \quad \mu_1(x) = \mathbb{E}[Y \mid X=x, T=1]
\]
\[
\hat{\tau}_{\text{T}}(x) = \mu_1(x) - \mu_0(x)
\]
*Limitation:* Does not share representation across groups; data efficiency drops by half.

### 2.3 TARNet (Treatment-Agnostic Representation Network)
Learns a shared latent representation \(\Phi(X) \in \mathbb{R}^p\) before branching into bifurcated hypothesis heads \(h_0\) and \(h_1\):
\[
\mathcal{L}_{\text{TARNet}} = \frac{1}{N} \sum_{i=1}^N \left[ (1 - T_i) \ell(Y_i, h_0(\Phi(X_i))) + T_i \ell(Y_i, h_1(\Phi(X_i))) \right]
\]

### 2.4 DragonNet with Targeted Regularization

DragonNet (Shi, Blei & Veitch, NeurIPS 2019) builds upon the sufficiency theorem for treatment effect estimation:

> **Theorem (Sufficiency):** For unconfounded CATE estimation, \(\Phi(X)\) needs only to preserve information relevant to both treatment assignment \(T\) and outcomes \(Y\).

DragonNet achieves this by:
1. Adding a **propensity prediction head** \(g(\Phi(X))\) trained with binary cross-entropy on \(T\).
2. Adding a scalar parameter \(\varepsilon\) and clever covariate \(H(t, x)\) from Targeted Maximum Likelihood Estimation (TMLE):

\[
H(t, x) \triangleq \frac{t}{g(\Phi(x))} - \frac{1 - t}{1 - g(\Phi(x))}
\]

The perturbed factual outcome prediction is:
\[
\tilde{Y}_i = \left( (1 - T_i) h_0(\Phi(X_i)) + T_i h_1(\Phi(X_i)) \right) + \varepsilon \cdot H(T_i, X_i)
\]

The complete training loss minimized by DragonNet is:
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{outcome}} + \alpha \mathcal{L}_{\text{propensity}} + \beta \mathcal{L}_{\text{targeted}}
\]
where \(\mathcal{L}_{\text{targeted}} = \frac{1}{N} \sum_{i=1}^N (Y_i - \tilde{Y}_i)^2\).

**Theoretical Guarantee:** Minimizing \(\mathcal{L}_{\text{targeted}}\) enforces the score equation of the average treatment effect to zero:
\[
\frac{1}{N} \sum_{i=1}^N \left( Y_i - \tilde{Y}_i \right) H(T_i, X_i) = 0
\]
This ensures double robustness and achieves the non-parametric efficiency bound for treatment effect estimation.

---

## 3. Four-Cohort Behavioral Matrix

```
                          Control Outcome (Y^(0))
                         Doesn't Buy          Buys
                   ┌───────────────────┬───────────────────┐
              Buys │  🟢 PERSUADABLE   │   ⚪ SURE THING   │
Treated            │  (Target: pure    │  (Skip: wasted    │
Outcome            │   incremental     │   margin, buys    │
(Y^(1))            │   margin lift)    │   anyway)         │
                   ├───────────────────┼───────────────────┤
       Doesn't Buy │  ⚫ LOST CAUSE    │  🔴 SLEEPING DOG  │
                   │  (Skip: zero      │  (Never Target:   │
                   │   response to     │   promo backfires/│
                   │   marketing)      │   causes churn)   │
                   └───────────────────┴───────────────────┘
```

---

## 4. Evaluation Metrics Formulation

### 4.1 Qini Curve
Candidates ranked descending by predicted CATE \(\hat{\tau}(X)\). At rank \(k \in \{1, \dots, N\}\):
\[
Q(k) = Y_{T, k} - Y_{C, k} \cdot \left( \frac{N_{T, k}}{N_{C, k}} \right)
\]
Random baseline connects origin \((0, 0)\) to \(Q(N)\):
\[
Q_{\text{rand}}(k) = \frac{k}{N} \cdot \left( Y_{T, \text{tot}} - Y_{C, \text{tot}} \cdot \frac{N_{T, \text{tot}}}{N_{C, \text{tot}}} \right)
\]
The Qini coefficient is the area between the two curves:
\[
Q = \int_0^1 \left( Q(p) - Q_{\text{rand}}(p) \right) dp
\]

### 4.2 AUUC (Area Under Uplift Curve)
Cumulative uplift difference weighted by cohort size:
\[
U(k) = \left( \frac{Y_{T, k}}{N_{T, k}} - \frac{Y_{C, k}}{N_{C, k}} \right) \cdot (N_{T, k} + N_{C, k})
\]

### 4.3 Off-Policy Policy Value (IPW)
For a marketing policy \(\pi(X) \in \{0, 1\}\):
\[
\hat{V}_{\text{IPW}}(\pi) = \frac{1}{N} \sum_{i=1}^N \frac{\mathbb{I}(T_i = \pi(X_i)) Y_i}{P(T_i = \pi(X_i) \mid X_i)}
\]

---

## 5. Autonomous Retail AI Mesh Integration

Uplift-DRL forms the causal core of the quick-commerce operating architecture:

1. **Upstream Ingestion:** Ingests candidate promo depth and markdown elasticities from `Multi-Agent-Dynamic-Pricing-Swarm` and `Perishable-Markdown-Optimizer`.
2. **Causal Audience Scoring:** Scores the entire active customer base, excluding Sleeping Dogs and non-responders.
3. **Downstream RecSys Amplification:** Passes the selected high-uplift Persuadable customer IDs to `BasketGNN`, which boosts inventory-aware cross-sell items specifically for shoppers primed to convert.
4. **Autonomous P&L Feedback:** Feeds campaign-level conversion lift back into commercial planning dashboards.
