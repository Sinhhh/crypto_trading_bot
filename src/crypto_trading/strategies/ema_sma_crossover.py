"""EMA/SMA cross signal.

Exposes only: `ema_sma_cross_signal(...)` -> "BUY" / "SELL" / "HOLD".

This is a minimal moving-average crossover signal:
- BUY when fast EMA crosses above slow SMA.
- SELL when fast EMA crosses below slow SMA.

Optional filters:
- `require_price_confirmation`: also require close to be above (BUY) or below (SELL)
  the slow SMA on the signal candle.
- `min_separation`: require a minimum relative separation between the two averages
  after the cross (e.g. 0.001 = 0.1%).
"""

from __future__ import annotations

import pandas as pd

from crypto_trading.indicators.moving_averages import ema, sma


def ema_sma_cross_signal(
    df_1h: pd.DataFrame,
    *,
    fast_ema: int = 12,
    slow_sma: int = 26,
    require_price_confirmation: bool = True,
    min_separation: float = 0.0,
) -> str:
    """EMA/SMA cross signal.

    Returns only: "BUY" / "SELL" / "HOLD".

    BUY:
      - EMA(fast) crosses above SMA(slow)
      - optionally close >= SMA(slow)

    SELL:
      - EMA(fast) crosses below SMA(slow)
      - optionally close <= SMA(slow)
    """
    if df_1h is None or len(df_1h) < max(int(fast_ema), int(slow_sma), 3) + 2:
        return "HOLD"

    if "close" not in df_1h.columns:
        return "HOLD"

    df = df_1h.copy()
    close = df["close"].astype(float)

    fast = int(fast_ema)
    slow = int(slow_sma)
    if fast <= 0 or slow <= 0:
        return "HOLD"

    df["ema_fast"] = ema(close, fast)
    df["sma_slow"] = sma(close, slow)

    prev = df.iloc[-2]
    last = df.iloc[-1]

    if pd.isna(prev[["ema_fast", "sma_slow"]]).any() or pd.isna(last[["ema_fast", "sma_slow"]]).any():
        return "HOLD"

    ema_prev = float(prev["ema_fast"])
    sma_prev = float(prev["sma_slow"])
    ema_last = float(last["ema_fast"])
    sma_last = float(last["sma_slow"])
    c_last = float(last["close"])

    # Optional post-cross separation filter
    sep_ok = True
    if float(min_separation) > 0.0:
        denom = abs(sma_last)
        if denom == 0.0:
            return "HOLD"
        sep_ok = abs(ema_last - sma_last) / denom >= float(min_separation)

    if not sep_ok:
        return "HOLD"

    crossed_up = ema_prev <= sma_prev and ema_last > sma_last
    crossed_down = ema_prev >= sma_prev and ema_last < sma_last

    if crossed_up:
        if require_price_confirmation and not (c_last >= sma_last):
            return "HOLD"
        return "BUY"

    if crossed_down:
        if require_price_confirmation and not (c_last <= sma_last):
            return "HOLD"
        return "SELL"

    return "HOLD"
