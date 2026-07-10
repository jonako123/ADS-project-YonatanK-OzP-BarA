"""Shared figure styling and small plot helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = "colorblind"


def set_style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE, context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })


def annotated_heatmap(matrix: pd.DataFrame, title: str, fmt: str = ".2f",
                      cmap: str = "vlag", center: float | None = 0.0,
                      figsize: tuple = (9, 7), mask_upper: bool = False):
    """Heatmap with sensible defaults for association matrices."""
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(matrix, dtype=bool)) if mask_upper else None
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, center=center,
                square=True, linewidths=0.5, mask=mask,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def barh_effect_sizes(s: pd.Series, title: str, xlabel: str,
                      thresholds: dict[str, float] | None = None,
                      figsize: tuple = (8, None)):
    """Horizontal bar chart of a sorted effect-size Series, with optional
    reference lines (e.g. small/medium/large conventions)."""
    s = s.sort_values()
    h = max(2.5, 0.4 * len(s) + 1)
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] or h))
    ax.barh(s.index.astype(str), s.values, color=sns.color_palette(PALETTE)[0])
    if thresholds:
        for name, v in thresholds.items():
            ax.axvline(v, ls="--", lw=1, color="grey")
            ax.text(v, ax.get_ylim()[1], f" {name}", va="top", fontsize=8, color="grey")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
