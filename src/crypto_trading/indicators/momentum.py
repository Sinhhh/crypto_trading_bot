"""Momentum indicators.

Consolidated indicator module:
- `rsi` (Relative Strength Index)
- `adx` (Average Directional Index)

These are simple implementations designed for lightweight strategy use.
"""

from __future__ import annotations

import pandas as pd


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute a simplified RSI over a given period.

    Returns values in [0, 100].
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=int(period)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=int(period)).mean()
    avg_gain = gain.rolling(int(period)).mean()
    avg_loss = loss.rolling(int(period)).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX over the specified period.

    Assumes input DataFrame with columns: high, low, close.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    plus_di = 100 * (plus_dm.rolling(int(period)).mean() / tr.rolling(int(period)).mean())
    minus_di = 100 * (
        minus_dm.rolling(int(period)).mean() / tr.rolling(int(period)).mean()
    )

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(int(period)).mean()
