"""Relative Strength Index (RSI) indicator.

Implements a simple RSI approximation using rolling means. For trading,
this serves as a lightweight signal for oversold/overbought regimes.
"""
# RSI indicator
import pandas as pd


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute a simplified RSI over a given period.

    Parameters:
    - series: Price series (e.g., close).
    - period: Lookback period for RSI.

    Returns: RSI values in [0, 100].
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
