"""Moving average helpers.

Consolidated indicator module:
- `ema` (Exponential Moving Average)
- `sma` (Simple Moving Average)
- `wma` (Weighted Moving Average)

These are lightweight pandas-based implementations intended for strategy logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Compute the Exponential Moving Average (EMA)."""
    return series.ewm(span=int(period), adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Compute the Simple Moving Average (SMA)."""
    return series.rolling(window=int(period), min_periods=1).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Compute the Weighted Moving Average (WMA) with linear weights."""

    def weighted_avg(x: np.ndarray) -> float:
        n = len(x)
        if n == 0:
            return float("nan")
        weights = np.arange(1, n + 1, dtype=float)
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window=int(period), min_periods=1).apply(weighted_avg, raw=True)
