"""Market regime detection for switching strategies.

Classifies market conditions into:
- TREND_UP / TREND_DOWN
- RANGE
- CROSS_UP / CROSS_DOWN (fresh EMA/SMA crossover when trend is not yet confirmed)
- TRANSITION

Uses ADX, moving-average slope, and Bollinger Band width, with an additional
EMA(12)/SMA(26) crossover fallback.
"""
import pandas as pd

from crypto_trading.indicators.momentum import adx
from crypto_trading.indicators.volatility import bollinger_width
from crypto_trading.indicators.moving_averages import sma, ema


def _ema_sma_cross_regime(
    df: pd.DataFrame,
    *,
    fast_ema: int = 12,
    slow_sma: int = 26,
) -> str | None:
    if df is None or "close" not in df.columns:
        return None
    if len(df) < max(int(fast_ema), int(slow_sma), 3) + 2:
        return None

    close = df["close"].astype(float)
    ema_fast = ema(close, int(fast_ema))
    sma_slow = sma(close, int(slow_sma))

    prev_fast = float(ema_fast.iloc[-2])
    prev_slow = float(sma_slow.iloc[-2])
    last_fast = float(ema_fast.iloc[-1])
    last_slow = float(sma_slow.iloc[-1])
    if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(last_fast) or pd.isna(last_slow):
        return None

    crossed_up = prev_fast <= prev_slow and last_fast > last_slow
    crossed_down = prev_fast >= prev_slow and last_fast < last_slow
    if crossed_up:
        return "CROSS_UP"
    if crossed_down:
        return "CROSS_DOWN"
    return None


def detect_regime(df: pd.DataFrame) -> str:
    """
    Detect market regime: 'TREND_UP', 'TREND_DOWN', 'RANGE', 'CROSS_UP',
    'CROSS_DOWN', or 'TRANSITION'.

    Heuristics:
    - TREND_UP if ADX > 25, EMA50 slope > 0, EMA50 > SMA200
    - TREND_DOWN if ADX > 25, EMA50 slope < 0, EMA50 < SMA200
    - RANGE if ADX < 20 and BB width in the bottom 20% of last 100 periods
    - CROSS_UP / CROSS_DOWN if an EMA12/SMA26 cross just occurred
    - Otherwise TRANSITION.

    Input must include 'high', 'low', 'close' columns.
    """
    if df is None or len(df) < 200:
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
    cross = _ema_sma_cross_regime(df)
    if cross is not None:
        return cross

    return "TRANSITION"
