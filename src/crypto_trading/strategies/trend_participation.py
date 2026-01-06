"""Trend participation (pullback) signal.

Exposes only: `trend_participation_signal(...)` → "BUY" / "SELL" / "HOLD".
"""

import pandas as pd

from crypto_trading.indicators.moving_averages import ema, wma
from crypto_trading.indicators.momentum import rsi


def trend_participation_signal(
    df_1h: pd.DataFrame,
    regime: str,  # "TREND_UP" or "TREND_DOWN"
    pullback_ma: str = "wma",  # "wma" or "ema"
    rsi_low: float = 40.0,
    rsi_high: float = 60.0,
    tag_tolerance: float = 0.0,
) -> str:
    """Trend Participation (Pullback) Strategy

    Returns only: "BUY" / "SELL" / "HOLD".

    Long (TREND_UP):
      - close > EMA50
      - low <= MA20 * (1 + tol)
      - close > MA20
      - RSI in [rsi_low, rsi_high]

    Short (TREND_DOWN) mirror:
      - close < EMA50
      - high >= MA20 * (1 - tol)
      - close < MA20
      - RSI in [rsi_low, rsi_high]
    """
    if regime not in ("TREND_UP", "TREND_DOWN"):
        return "HOLD"

    if df_1h is None or len(df_1h) < 60:
        return "HOLD"

    if not {"high", "low", "close"}.issubset(df_1h.columns):
        return "HOLD"

    df = df_1h.copy()
    close = df["close"].astype(float)

    if pullback_ma == "wma":
        df["ma20"] = wma(close, 20)
    else:
        df["ma20"] = ema(close, 20)

    df["ema50"] = ema(close, 50)
    df["rsi14"] = rsi(close, 14)

    last = df.iloc[-1]
    if pd.isna(last[["ma20", "ema50", "rsi14"]]).any():
        return "HOLD"

    ma20 = float(last["ma20"])
    ema50 = float(last["ema50"])
    r = float(last["rsi14"])
    c = float(last["close"])
    lo = float(last["low"])
    hi = float(last["high"])

    if not (float(rsi_low) <= r <= float(rsi_high)):
        return "HOLD"

    tol = float(tag_tolerance)

    if regime == "TREND_UP":
        pullback_long = c > ema50 and lo <= ma20 * (1.0 + tol) and c > ma20
        return "BUY" if pullback_long else "HOLD"

    pullback_short = c < ema50 and hi >= ma20 * (1.0 - tol) and c < ma20
    return "SELL" if pullback_short else "HOLD"
