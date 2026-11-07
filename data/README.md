# Uplift-DRL Datasets

This directory houses raw and processed benchmarks for uplift modeling and Conditional Average Treatment Effect (CATE) estimation.

---

## 1. Quick-Commerce Synthetic Dataset (Immediate / Zero Download)

Generate an instant, stratified, benchmark dataset with ground-truth treatment assignments, counterfactual outcomes, and four customer cohorts (Persuadables, Sure Things, Lost Causes, Sleeping Dogs):

```bash
python -m src.data --source synthetic --n-samples 25000 --output data/processed/synthetic.npz
```

This creates `data/processed/synthetic.npz` with stratified 70/15/15 train/val/test splits, feature scalers, and ground-truth ITE labels.

---

## 2. Kevin Hillstrom MineThatData Email Analytics Benchmark

The classic e-mail marketing A/B testing dataset (64,000 customers).

- **Source:** [MineThatData E-Mail Analytics](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
- **Direct CSV Download:**
  ```bash
  curl -o data/hillstrom.csv https://raw.githubusercontent.com/klarsen1/MineThatData/master/MineThatData.csv
  ```
- **Treatment Arms:** `Mens E-Mail` (T=1), `Womens E-Mail` (T=1), `No E-Mail` (T=0).
- **Target Variables:** `visit` (0/1), `conversion` (0/1), `spend` (continuous float).
- **Processing Command:**
  ```bash
  python -m src.data --source hillstrom --input data/hillstrom.csv --output data/processed/hillstrom.npz
  ```

---

## 3. Criteo Uplift Prediction Dataset v2

The large-scale digital advertising standard (13.9M rows, 12 anonymized continuous features).

- **Source:** [Criteo AI Lab Uplift Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- **Download:** Download `criteo-uplift-v2.1.csv.gz` from Criteo AI Lab, decompress to `data/criteo.csv`.
- **Columns:** `f0`-`f11`, `exposure` (treatment), `visit`, `conversion`.
- **Processing Command:**
  ```bash
  python -m src.data --source criteo --input data/criteo.csv --output data/processed/criteo.npz --subsample 500000
  ```

---

## 4. Starbucks Mobile Promo Dataset

An A/B test simulating customer reaction to mobile discount coupons.

- **Source:** [Starbucks Promo on Kaggle](https://www.kaggle.com/datasets/snowman6210/starbucks)
- **Processing Command:**
  ```bash
  python -m src.data --source starbucks --input data/starbucks.csv --output data/processed/starbucks.npz
  ```

---

## Processed `.npz` Archive Structure

All loaders produce a standardized `.npz` archive consumed by `src.train` and `src.eval`:
- `X_train`, `X_val`, `X_test`: Normalized feature tensors (`float32`, shape `[N, D]`)
- `T_train`, `T_val`, `T_test`: Binary treatment indicator (`float32`, shape `[N, 1]`)
- `Y_train`, `Y_val`, `Y_test`: Binary observed outcome (`float32`, shape `[N, 1]`)
- `feature_names`: Array of string column names
- `true_ite_test` (optional for synthetic): Known counterfactual \(Y^{(1)} - Y^{(0)}\) for evaluation
