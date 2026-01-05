"""ATR and Bollinger Band utilities.

Provides ATR, Bollinger Bands (mid/upper/lower), and band width for
range/trend assessment and risk management.
"""
import pandas as pd

from indicators.ma import sma


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR).

    Parameters:
    - df: DataFrame with high, low, close.
    - period: Lookback period.

    Returns: ATR series.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bollinger_bands(series: pd.Series, period: int = 20, std_multiplier: float = 2.0):
    """Compute Bollinger Bands (mid, upper, lower).

    Returns a tuple of (mid, upper, lower) series.
    """
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_multiplier * std
    lower = mid - std_multiplier * std
    return mid, upper, lower

def bollinger_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute the absolute width of Bollinger Bands for a close series."""
    mid = sma(df["close"], period)
    std = df["close"].rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = upper - lower
    return width
