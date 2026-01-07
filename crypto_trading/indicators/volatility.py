"""Volatility indicators.

Consolidated indicator module:
- `atr` (Average True Range)
- `bollinger_bands` (mid/upper/lower)
- `bollinger_width` (absolute band width)
"""

from __future__ import annotations

import pandas as pd

from crypto_trading.indicators.moving_averages import sma


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR)."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(int(period)).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (mid, upper, lower)."""
    mid = sma(series, int(period))
    std = series.rolling(int(period)).std()
    upper = mid + float(std_multiplier) * std
    lower = mid - float(std_multiplier) * std
    return mid, upper, lower


def bollinger_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute absolute Bollinger Band width for df['close']."""
    mid = sma(df["close"], int(period))
    std = df["close"].rolling(int(period)).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper - lower
