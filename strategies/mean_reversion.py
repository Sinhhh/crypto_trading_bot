"""
Mean-reversion strategy using RSI and Bollinger Bands.

When RSI is oversold and price tags the lower Bollinger Band, generate a
long entry with ATR-based stop and mid-band target. Otherwise, hold.
"""
from dataclasses import dataclass

import pandas as pd

from indicators.rsi import rsi
from indicators.bb_atr import atr, bollinger_bands


@dataclass
class StrategySignal:
    """Strategy decision payload for downstream execution/reporting."""
    action: str  # "enter_long" or "hold"
    entry: float | None
    stop_loss: float | None
    take_profit: float | None
    confidence: float
    reason: str


def mean_reversion_strategy(df_1h: pd.DataFrame) -> StrategySignal:
    """
    Compute a mean-reversion signal from 1h candles.

    Conditions:
    - Enter long when RSI < 35 and close <= lower Bollinger Band.
    - Place stop at 1x ATR below entry and target at mid band.

    Returns a `StrategySignal` with `action` of "enter_long" or "hold".
    """
    if len(df_1h) < 20:
        return StrategySignal("hold", None, None, None, 0.0, "Insufficient data")

    df = df_1h.copy()
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df)
    mid, upper, lower = bollinger_bands(df["close"], 20)
    df["bb_mid"] = mid
    df["bb_low"] = lower

    latest = df.iloc[-1]
    if not (
        pd.notna(latest["rsi"])
        and pd.notna(latest["bb_low"])
        and pd.notna(latest["bb_mid"])
        and pd.notna(latest["atr"])
    ):
        return StrategySignal("hold", None, None, None, 0.0, "Indicators not ready")

    if not (latest["rsi"] < 35 and latest["close"] <= latest["bb_low"]):
        return StrategySignal("hold", None, None, None, 0.0, "No mean reversion setup")

    entry = float(latest["close"])
    atr_mult = 1.0
    stop = entry - atr_mult * float(latest["atr"])
    target = float(latest["bb_mid"])

    if stop >= entry or target <= entry or latest["atr"] <= 0:
        return StrategySignal("hold", None, None, None, 0.0, "Invalid risk setup")

    return StrategySignal(
        action="enter_long",
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        confidence=0.65,
        reason="RSI oversold + price at lower Bollinger Band",
    )
