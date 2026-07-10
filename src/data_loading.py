"""Data acquisition and assembly for the chocolate-sales EDA.

Single source of truth for loading so the notebook stays narrative-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_SLUG = "ssssws/chocolate-sales-dataset-2023-2024"
TABLE_STEMS = ("sales", "products", "stores", "customers")


def download_dataset() -> Path:
    """Resolve the dataset directory.

    Priority: CHOCO_DATA_DIR env var (offline/dev override) -> kagglehub
    download (uses local cache when available).
    """
    import os

    override = os.environ.get("CHOCO_DATA_DIR")
    if override and Path(override).exists():
        return Path(override)

    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_SLUG))


def load_tables(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the four raw tables as a dict of DataFrames."""
    data_dir = Path(data_dir) if data_dir else download_dataset()
    return {s: pd.read_csv(data_dir / f"{s}.csv") for s in TABLE_STEMS}


def load_calendar(data_dir: Path | None = None) -> pd.DataFrame | None:
    """Load the auxiliary calendar table if present (v2 of the dataset)."""
    data_dir = Path(data_dir) if data_dir else download_dataset()
    p = data_dir / "calendar.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def referential_integrity(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count sales rows whose foreign keys are missing from each dimension."""
    s = tables["sales"]
    rows = []
    for dim, key in (("products", "product_id"), ("stores", "store_id"), ("customers", "customer_id")):
        orphan = ~s[key].isin(tables[dim][key])
        rows.append({"dimension": dim, "key": key,
                     "orphan_rows": int(orphan.sum()),
                     "orphan_pct": float(orphan.mean() * 100),
                     "distinct_orphan_keys": int(s.loc[orphan, key].nunique())})
    return pd.DataFrame(rows)


def build_merged(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Star-schema join -> one analysis-ready transaction table.

    - inner join on product (drops orphan product_ids — quantified separately)
    - parses dates, adds calendar columns and derived economics
    """
    df = (
        tables["sales"]
        .merge(tables["products"], on="product_id", how="inner")
        .merge(tables["stores"], on="store_id", how="inner")
        .merge(tables["customers"], on="customer_id", how="inner")
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["join_date"] = pd.to_datetime(df["join_date"])

    # calendar
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["weekday"] = df["order_date"].dt.dayofweek          # 0 = Monday
    df["is_weekend"] = df["weekday"] >= 5

    # derived economics (the analysis targets; raw revenue/cost/profit are identities)
    df["margin"] = np.where(df["revenue"] > 0, df["profit"] / df["revenue"], np.nan)
    df["effective_price"] = df["revenue"] / df["quantity"]
    df["unit_cost"] = df["cost"] / df["quantity"]
    df["is_discounted"] = df["discount"] > 0
    df["customer_tenure_days"] = (df["order_date"] - df["join_date"]).dt.days
    return df


def identity_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Verify the accounting identities and return residual summaries.

    revenue ?= quantity * unit_price * (1 - discount)
    profit  ?= revenue - cost
    """
    r1 = df["revenue"] - df["quantity"] * df["unit_price"] * (1 - df["discount"])
    r2 = df["profit"] - (df["revenue"] - df["cost"])
    out = pd.DataFrame({
        "identity": ["revenue = qty*price*(1-disc)", "profit = revenue - cost"],
        "max_abs_residual": [float(r1.abs().max()), float(r2.abs().max())],
        "share_within_1_cent": [float((r1.abs() <= 0.01).mean()), float((r2.abs() <= 0.01).mean())],
    })
    return out
