"""Moving average helpers (EMA, SMA, WMA).

These utilities provide lightweight moving-average implementations using pandas,
intended for feature calculation and strategy logic.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Compute the Exponential Moving Average (EMA).

    Parameters:
    - series: Price series (e.g., close).
    - period: Lookback period.

    Returns: EMA series aligned to input index.
    """
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Compute the Simple Moving Average (SMA).

    Parameters:
    - series: Price series (e.g., close).
    - period: Lookback period.

    Returns: SMA series aligned to input index.
    """
    return series.rolling(window=period, min_periods=1).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Compute the Weighted Moving Average (WMA).

    Parameters:
    - series: Price series (e.g., close).
    - period: Lookback period.

    Returns: WMA series aligned to input index.

    Uses linearly increasing weights over the rolling window.

    Notes:
    - Works with `min_periods=1` by scaling weights to the actual window length.
    """

    def weighted_avg(x: np.ndarray) -> float:
        n = len(x)
        if n == 0:
            return float("nan")
        weights = np.arange(1, n + 1, dtype=float)
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window=period, min_periods=1).apply(weighted_avg, raw=True)
