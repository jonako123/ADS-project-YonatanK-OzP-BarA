# EDA.ipynb — Plan of Action (Phase 1)

**Dataset:** Kaggle `ssssws/chocolate-sales-dataset-2023-2024` — 1M transactions × 4 dimension tables (products 200, stores 100, customers 50K), 2023–2024.
**Sources ingested:** FAQ doc, PROJECT_PLAYBOOK.md, baseline churn notebook, current EDA.ipynb (25 cells).

---

## 1. Requirements map (FAQ → notebook)

| FAQ requirement | Where it is satisfied |
|---|---|
| Real-world (not synthetic) dataset | ⚠️ **Risk — see §2.** Addressed head-on by a data-forensics section |
| "Various insights" | §4–§10: association study, hypothesis battery, segmentation, time series |
| Prediction models | Phase 3 (separate notebook); EDA ends with candidate targets + modeling brief |
| Presentation: problem definition / data analysis / model performance | Notebook sections 1, 4–10, and Phase 3 respectively |
| Python | Yes; data itself excluded from submission (already gitignored) |

The FAQ is deliberately high-level; the binding quality bar is PROJECT_PLAYBOOK.md (association strength beyond Pearson, distribution reading with intent, ablation-proven features, honest reporting). The plan below implements that bar.

## 2. Critical findings about the current baseline (why we must go deeper)

1. **All 9 "strong" Pearson correlations are accounting identities.** `revenue ≈ quantity × unit_price × (1−discount)`, `profit = revenue − cost`. They are tautologies, not insights — a grader will flag this immediately. Fix: analyze **derived economics** (profit margin, effective price, discount depth) and use **partial correlations** that control for quantity/price.
2. **Suspected synthetic data.** Perfectly clean 1M rows, zero missing values, and city/country mismatches (Sydney→UK, New York→Australia). The FAQ *requires* non-synthetic data. Plan: run a formal forensics section (Benford's law on revenue, uniformity/independence tests, calendar-structure tests) and report honestly. **Decision point for you:** keep this dataset (and defend/acknowledge it) or swap now while it's cheap.
3. **Leaky feature engineering.** Current target-encoding uses full-data means — must become out-of-fold before any modeling claims.
4. **FFT section is statistically naive.** No detrending/windowing, no significance threshold → the "top 20 periods" list is mostly spectral leakage around the true weekly peak. Replace with STL + ACF/PACF + significance-tested periodogram.
5. **n ≈ 10⁶ makes every p-value ≈ 0.** The notebook must lead with **effect sizes** (Cliff's δ, ε², Cramér's V, η²) and confidence intervals, not p-values — this is the single most important PhD-level framing for this dataset.

## 3. Target notebook structure (EDA.ipynb, ~12 sections)

**§0 Setup & reproducibility** — seeds, versions, style; helpers imported from `src/`.

**§1 Problem definition & analytical questions** — 5–6 explicit business questions (What drives profit margin? Is loyalty membership worth it? Are discounts profitable? Do store types/geographies differ? Is demand seasonal? Can we segment customers?). Each later section answers one.

**§2 Data acquisition, schema & integrity audit** — kagglehub load; ERD of the star schema; referential integrity (investigate the 9,764 orphan `product_id`s: which IDs, random or structured?); accounting-identity residual checks; dimension-consistency checks (city↔country).

**§3 Data forensics (synthetic-data assessment)** — Benford's law on revenue/cost; KS tests of order_date against uniform; independence of customer attributes; verdict reported honestly per playbook.

**§4 Univariate analysis with intent** — distributions + skewness/kurtosis, log-transform assessment for money columns; discount mass at 0 (zero-inflation → treat as "discounted? yes/no × depth"); age/tenure structure.

**§5 Derived economics** — `margin = profit/revenue`, `effective_price`, `discount_depth`, basket value. All downstream analysis runs on these, not the tautological raw columns.

**§6 Association study (the correlation deep-dive)** —
- Numeric×numeric: Pearson vs **Spearman** (divergence ⇒ nonlinearity), **partial correlations** controlling for quantity & unit_price, **distance correlation / mutual information** on key pairs.
- Categorical×numeric: **η² / Kruskal–Wallis ε²** (category, brand, store_type, country vs margin & revenue).
- Categorical×categorical: **Cramér's V** matrix (bias-corrected).
- One consolidated "association leaderboard" figure replacing the current |r|>0.1 scatter grid.

**§7 Hypothesis-testing battery** (all with effect sizes + BH-FDR correction):
- Loyalty vs non-loyalty spend/margin — Mann-Whitney + **Cliff's δ**.
- **Discount elasticity** — log-log quantity~discount regression; does the volume lift cover the margin give-away? (flagship insight)
- Cocoa % / weight vs price & margin — nonlinear check (LOWESS + Spearman).
- Store type & country margin differences — Kruskal + Dunn post-hoc.
- Age/gender effects — expect ≈ null; report honestly as negative results.

**§8 Time series** — STL decomposition of daily revenue/orders; weekday & month-of-year profiles with CIs; calendar heatmap; ACF/PACF; **properly-done spectral analysis** (detrended, Hann window, red-noise significance line); ADF/KPSS stationarity.

**§9 Multivariate structure** — **RFM customer segmentation** (K-means, k by silhouette; segment profiling); PCA on standardized product/store aggregates; category×store_type×country interaction heatmaps of margin.

**§10 Feature engineering + ablation** — tenure, out-of-fold target/frequency encodings, calendar features, interactions; each validated with a quick ablation vs a baseline regressor (playbook: "prove impact, don't assume").

**§11 Insight synthesis** — ranked table: insight → evidence (effect size, CI) → business action.

**§12 Bridge to modeling** — candidate targets (revenue/margin regression on log scale; loyalty classification; segment prediction), metric choices, leakage inventory.

## 4. Code architecture

- `EDA.ipynb` — narrative + figures only; every figure followed by a PhD-level interpretation paragraph (statistical meaning + business meaning).
- `src/data_loading.py` — kagglehub load, merge, dtype fixes, integrity checks.
- `src/stats_utils.py` — Cramér's V, η², Cliff's δ, partial corr, BH-FDR, bootstrap CIs.
- `src/plotting.py` — styled figure helpers.
- Heavy computations (bootstraps, clustering scans) cached to parquet so the notebook re-runs top-to-bottom fast.
- Environment gotchas from playbook respected (pandas 3.0 string dtype, `numpy<2.5`, `tick_labels=`).

## 5. Execution order (Phase 2, upon approval)

1. §0–§3 (foundation + forensics) → checkpoint with you on the synthetic-data verdict
2. §4–§6 (distributions + associations)
3. §7–§8 (hypotheses + time series)
4. §9–§10 (multivariate + features)
5. §11–§12 + polish, top-to-bottom re-run, README + requirements.txt

Each stage verified against executed output before moving on (playbook rule).
