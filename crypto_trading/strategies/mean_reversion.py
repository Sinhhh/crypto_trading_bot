"""
Mean-reversion (range) signal using RSI and Bollinger Bands.

Exposes only: `mean_reversion_signal(...)` → "BUY" / "SELL" / "HOLD".
"""

import pandas as pd

from crypto_trading.indicators.momentum import rsi
from crypto_trading.indicators.volatility import bollinger_bands


def mean_reversion_signal(
    df_1h: pd.DataFrame,
    regime: str,
    *,
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    tag_tolerance: float = 0.0,
) -> str:
    """Mean Reversion (Range) Signal

    Returns only: "BUY" / "SELL" / "HOLD".

    Only acts in `regime == "RANGE"`.

    Long (range bounce):
      - RSI <= rsi_oversold
      - low tags lower band (with tolerance)
      - close reclaims above lower band

    Short (range fade) mirror:
      - RSI >= rsi_overbought
      - high tags upper band (with tolerance)
      - close falls back below upper band
    """
    if regime != "RANGE":
        return "HOLD"

    if df_1h is None or len(df_1h) < max(int(bb_period) + 2, 30):
        return "HOLD"

    if not {"high", "low", "close"}.issubset(df_1h.columns):
        return "HOLD"

    df = df_1h.copy()
    close = df["close"].astype(float)
    df["rsi14"] = rsi(close, 14)
    mid, upper, lower = bollinger_bands(close, int(bb_period), float(bb_std))
    df["bb_mid"] = mid
    df["bb_up"] = upper
    df["bb_low"] = lower

    last = df.iloc[-1]
    if pd.isna(last[["rsi14", "bb_low", "bb_up"]]).any():
        return "HOLD"

    r = float(last["rsi14"])
    c = float(last["close"])
    lo = float(last["low"])
    hi = float(last["high"])
    bb_low = float(last["bb_low"])
    bb_up = float(last["bb_up"])
    tol = float(tag_tolerance)

    long_ok = r <= float(rsi_oversold) and lo <= bb_low * (1.0 + tol) and c > bb_low
    if long_ok:
        return "BUY"

    short_ok = r >= float(rsi_overbought) and hi >= bb_up * (1.0 - tol) and c < bb_up
    if short_ok:
        return "SELL"

    return "HOLD"
