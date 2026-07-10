"""Shared modelling setup so EDA sections and scripts use identical frames.

Honest feature policy (see §2.3 of the notebook): `quantity`, `revenue`,
`cost`, `profit` never appear on the feature side together with a money
target — they are linked to it by accounting identities, not by learnable
structure. `customer_tenure_days` is replaced by a cleaned version because
43.7% of raw tenures are negative (§3.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
NUM_FEATURES = ["unit_price", "discount", "cocoa_percent", "weight_g", "age",
                "loyalty_member", "tenure_clean"]
CAT_FEATURES = ["category", "brand", "store_type", "country", "gender"]


def prepare_modeling_data(df: pd.DataFrame, n_sample: int = 300_000,
                          seed: int = SEED) -> dict:
    """Sample, engineer honest features, split. Returns a dict of frames.

    - regression target: transaction revenue
    - classification target: high-value transaction (revenue >= global Q3)
    """
    sub = df.sample(n=min(n_sample, len(df)), random_state=seed).copy()
    # tenure with the impossible negative values floored at 0 + validity flag
    sub["tenure_clean"] = sub["customer_tenure_days"].clip(lower=0)

    X = sub[NUM_FEATURES + CAT_FEATURES]
    y_reg = sub["revenue"]
    q3 = df["revenue"].quantile(0.75)          # threshold from the full data
    y_clf = (sub["revenue"] >= q3).astype(int)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=0.25, random_state=seed, stratify=y_clf)

    return {"sub": sub, "q3": float(q3),
            "X_tr": X_tr, "X_te": X_te,
            "yr_tr": yr_tr, "yr_te": yr_te,
            "yc_tr": yc_tr, "yc_te": yc_te}
