from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load


# Keep the schema aligned with the existing ML gate feature set.
CATEGORICAL_FEATURES = ["symbol", "regime", "source"]
NUMERIC_FEATURES = [
    "bar_of_day",
    "day_of_week",
    "bars_to_utc_midnight",
    "close",
    "volume",
    "ret_1",
    "ret_3",
    "ret_6",
    "rv_20",
    "rsi14",
    "adx14",
    "dist_sma",
    "vol_ratio",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class VolatilityGate:
    model: Any

    def predict_proba_row(self, features: dict) -> float:
        """Return probability of y=1 (big move within horizon)."""
        df = pd.DataFrame([{k: features.get(k) for k in ALL_FEATURES}])
        proba = self.model.predict_proba(df)
        p1 = float(proba[0][1])
        if not np.isfinite(p1):
            return 0.0
        return max(0.0, min(1.0, p1))


def load_volatility_model(path: str | Path) -> VolatilityGate:
    p = Path(path)
    obj = load(p)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
    else:
        model = obj

    return VolatilityGate(model=model)
