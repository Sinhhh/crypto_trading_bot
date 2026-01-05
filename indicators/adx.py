"""
Average Directional Index (ADX) calculator.

Measures trend strength based on directional movement and true range.
Assumes input dataframe with high/low/close columns.
"""
import pandas as pd


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX over the specified period.

    Parameters:
    - df: DataFrame with columns high, low, close.
    - period: Lookback for smoothing and ADX calculation.

    Returns: ADX series where higher values indicate stronger trends.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

    plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()
