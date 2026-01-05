"""
Market regime detection for switching strategies.

Classifies market conditions into TREND, RANGE, or TRANSITION using
ADX, moving average slope, and Bollinger Band width.
"""
import pandas as pd

from indicators.adx import adx
from indicators.bb_atr import bollinger_width
from indicators.ma import sma, ema


def detect_regime(df: pd.DataFrame) -> str:
    """
    Detect market regime: 'TREND_UP', 'TREND_DOWN', 'RANGE', or 'TRANSITION'.

    Heuristics:
    - TREND_UP if ADX > 25, EMA50 slope > 0, EMA50 > SMA200
    - TREND_DOWN if ADX > 25, EMA50 slope < 0, EMA50 < SMA200
    - RANGE if ADX < 20 and BB width in the bottom 20% of last 100 periods
    - Otherwise TRANSITION.

    Input must include 'high', 'low', 'close' columns.
    """
    if len(df) < 200:
        return "TRANSITION"

    df = df.copy()
    # --- Indicators ---
    df["ema_50"] = ema(df["close"], 50)
    df["sma_200"] = sma(df["close"], 200)
    df["ema_slope"] = df["ema_50"].diff(5)  # EMA slope over last 5 bars
    df["adx"] = adx(df, 14)
    df["bb_width"] = bollinger_width(df, 20)

    latest = df.iloc[-1]

    # --- TREND Detection ---
    if latest["adx"] > 25:
        if latest["ema_slope"] > 0 and latest["ema_50"] > latest["sma_200"]:
            return "TREND_UP"
        elif latest["ema_slope"] < 0 and latest["ema_50"] < latest["sma_200"]:
            return "TREND_DOWN"

    # --- RANGE Detection ---
    bb_quantile = df["bb_width"].rolling(100, min_periods=20).quantile(0.2)
    if latest["adx"] < 20 and latest["bb_width"] < bb_quantile.iloc[-1]:
        return "RANGE"

    # --- TRANSITION / Weak Trend ---
    return "TRANSITION"
