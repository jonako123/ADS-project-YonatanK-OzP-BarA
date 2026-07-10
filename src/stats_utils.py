"""Statistical helpers for the chocolate-sales EDA.

Every function returns an *effect size* (with a p-value where relevant).
With n ~ 10^6, p-values are ~0 for even microscopic effects, so effect
sizes carry the scientific content; p-values are reported for completeness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- associations
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) for two categorical series.

    V in [0, 1]. Correction matters when the contingency table is large
    relative to n; without it V is inflated.
    """
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    phi2 = chi2 / n
    # bias correction
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """η (eta) — correlation ratio for categorical -> numeric association.

    η² = SS_between / SS_total, i.e. the share of the numeric variance
    explained by the grouping. Returns η (comparable scale to |r|).
    """
    df = pd.DataFrame({"g": categories, "v": values}).dropna()
    grand = df["v"].mean()
    ss_total = ((df["v"] - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    agg = df.groupby("g", observed=True)["v"].agg(["count", "mean"])
    ss_between = (agg["count"] * (agg["mean"] - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta via the Mann-Whitney U relation: δ = 2U/(n1 n2) − 1.

    O(n log n); safe for millions of rows. |δ| < .147 negligible,
    < .33 small, < .474 medium, else large (Romano et al. 2006).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def kruskal_epsilon_sq(values: pd.Series, groups: pd.Series) -> tuple[float, float, float]:
    """Kruskal-Wallis H test + ε² effect size. Returns (H, p, epsilon_sq).

    ε² = (H − k + 1) / (n − k): the rank-based analogue of η².
    """
    df = pd.DataFrame({"g": groups, "v": values}).dropna()
    samples = [g["v"].to_numpy() for _, g in df.groupby("g", observed=True)]
    h, p = stats.kruskal(*samples)
    n, k = len(df), len(samples)
    eps2 = max(0.0, (h - k + 1) / (n - k))
    return float(h), float(p), float(eps2)


def partial_corr(df: pd.DataFrame, x: str, y: str, covars: list[str]) -> float:
    """Partial Pearson correlation of x and y controlling for `covars`,
    computed as the correlation of OLS residuals.
    """
    sub = df[[x, y] + covars].dropna().to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(sub)), sub[:, 2:]])
    beta_x, *_ = np.linalg.lstsq(X, sub[:, 0], rcond=None)
    beta_y, *_ = np.linalg.lstsq(X, sub[:, 1], rcond=None)
    rx = sub[:, 0] - X @ beta_x
    ry = sub[:, 1] - X @ beta_y
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ inference
def bh_fdr(pvals: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg adjusted p-values (q-values) + rejection flags."""
    p = pvals.to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank down
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return pd.DataFrame({"p": p, "q_bh": out, "reject": out < alpha}, index=pvals.index)


def bootstrap_ci(
    values: np.ndarray,
    stat=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = np.array([stat(rng.choice(values, size=len(values), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(stat(values)), float(lo), float(hi)


# ------------------------------------------------------------------ forensics
def benford_first_digit(values: np.ndarray) -> pd.DataFrame:
    """Observed vs Benford-expected first-digit frequencies + chi² distance.

    Real multi-scale financial amounts tend to follow Benford's law;
    uniformly simulated prices do not. A diagnostic, not a proof.
    """
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    first = (v / 10 ** np.floor(np.log10(v))).astype(int)
    obs = pd.Series(first).value_counts(normalize=True).sort_index()
    obs = obs.reindex(range(1, 10), fill_value=0.0)
    exp = pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)})
    mad = float((obs - exp).abs().mean())  # Nigrini's MAD criterion
    return pd.DataFrame({"observed": obs, "benford": exp, "abs_diff": (obs - exp).abs()}).assign(mad=mad)
