# Chocolate Sales 2023–2024 — Applied DS Final Project (EDA + Models)

**Team:** Yonatan K · Oz P · Bar A — Applied Data Science, Bar-Ilan University

## Dataset

[Chocolate Sales Dataset 2023–2024](https://www.kaggle.com/datasets/ssssws/chocolate-sales-dataset-2023-2024)
(Kaggle, v2): 1M transactions × 200 products × 100 stores × 50K customers, star schema.
Downloaded programmatically via `kagglehub` — **no data files are committed** (course rule).

## Structure

- `EDA.ipynb` — the full analysis: integrity audit & data forensics (§2–§3), distributions (§4),
  derived economics (§5), association study (§6), hypothesis battery with effect sizes + BH-FDR
  (§7), time series with permutation-tested spectrum (§8), RFM segmentation (§9), ablation-proven
  feature engineering incl. a leakage post-mortem (§10), ranked insights (§11), and predictive
  models with a full evaluation suite (§13: regression vs theoretical ceiling; calibrated
  classifier with ROC/PR/threshold/gains; negative-control model).
- `src/data_loading.py` — acquisition, star-schema assembly, integrity checks
- `src/stats_utils.py` — Cramér's V, η, Cliff's δ, ε², partial correlation, BH-FDR, bootstrap, Benford
- `src/plotting.py` — figure styling helpers
- `src/modeling.py` — shared leak-free modelling setup

## How to run

```bash
pip install -r requirements.txt
jupyter lab EDA.ipynb   # runs top-to-bottom; first run downloads the data via kagglehub
```

All randomness is fixed (`random_state=42`).

## Headline findings

1. **Discounts are pure loss** — demand elasticity β = −0.000 (CI ±0.013); ≈ 601K profit foregone (+6%).
2. **Margin is exogenous U(0.30, 0.50) noise** — no margin structure exists on any dimension.
3. All demographic/geographic/channel/calendar effects are **precisely bounded nulls** (16 tests, 0 survive FDR).
4. Revenue is predictable exactly to its **derived theoretical ceiling** (R² 0.376/0.377); high-value
   transaction classifier reaches ROC-AUC 0.84 with calibrated probabilities.
5. The dataset is demonstrably **synthetic** (5 independent forensic diagnostics, §3) — used with
   course approval; findings characterise methodology and the generator, not the chocolate market.
