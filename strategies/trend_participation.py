"""
Trend participation strategy using multi-timeframe confirmation.

Overview:
- 4h trend permission: EMA(50) above/below EMA(200) with positive/negative
  EMA(50) slope over the last 5 intervals → "UP"/"DOWN"; else "NONE".
- 1h entry: In an UP trend, enter on pullback when price tags/undershoots
  EMA(20), remains above EMA(50), and RSI is neutral (40–60).

Outputs a `StrategySignal` with ATR-based stop and a 3×ATR target.
"""
import pandas as pd

from dataclasses import dataclass

from indicators.bb_atr import atr
from indicators.ma import ema, wma
from indicators.rsi import rsi


@dataclass
class StrategySignal:
    """Decision payload for downstream execution/reporting.

    Attributes:
    - action: String directive, e.g., "enter_long" or "hold".
    - entry: Entry price if applicable, else None.
    - stop_loss: Protective stop level if applicable, else None.
    - take_profit: Target price if applicable, else None.
    - confidence: Heuristic confidence in [0,1].
    - reason: Human-readable rationale.
    """
    action: str
    entry: float | None
    stop_loss: float | None
    take_profit: float | None
    confidence: float
    reason: str

def detect_trend_permission(df_4h: pd.DataFrame) -> str:
    """Detect trend direction using EMA(50/200) on 4h candles.

    Returns:
    - "UP" when EMA50 > EMA200 and EMA50 slope (5-interval diff) > 0.
    - "DOWN" when EMA50 < EMA200 and slope < 0.
    - "NONE" otherwise or if insufficient data.
    """
    df = df_4h.copy()
    if len(df) < 210:
        return "NONE"
    df["ema_50"] = ema(df["close"], 50)
    df["ema_200"] = ema(df["close"], 200)
    slope = df["ema_50"].diff(5)

    latest = df.iloc[-1]
    if pd.isna(latest["ema_50"]) or pd.isna(latest["ema_200"]) or pd.isna(slope.iloc[-1]):
        return "NONE"

    if latest["ema_50"] > latest["ema_200"] and slope.iloc[-1] > 0:
        return "UP"
    if latest["ema_50"] < latest["ema_200"] and slope.iloc[-1] < 0:
        return "DOWN"
    return "NONE"

def trend_participation_strategy(df_1h: pd.DataFrame, trend_permission: str, pullback_ma: str = "wma") -> StrategySignal:
    """Enter long on 1h pullbacks during a confirmed 4h uptrend.

    Preconditions:
    - `trend_permission` must be "UP".
    - `df_1h` must contain OHLC columns and at least ~50 rows.

    Entry filter (latest candle):
    - `close` > EMA50 (trend integrity)
    - `low` ≤ WMA20 (pullback tag)
    - 40 ≤ RSI(14) ≤ 60 (neutral reset)

    Returns a `StrategySignal` with action "enter_long" or "hold".
    """
    if trend_permission != "UP":
        return StrategySignal("hold", None, None, None, 0.0, "Trend not allowed or no trend")

    df = df_1h.copy()
    if len(df) < 50 or not {"high", "low", "close"}.issubset(df.columns):
        return StrategySignal("hold", None, None, None, 0.0, "Insufficient data or missing OHLC")

    if pullback_ma == "wma":
        df["pb_20"] = wma(df["close"], 20)
    else:
        df["pb_20"] = ema(df["close"], 20)

    df["ema_50"] = ema(df["close"], 50)
    df["atr"] = atr(df, 14)
    df["rsi"] = rsi(df["close"], 14)

    last = df.iloc[-1]
    if pd.isna(last[["pb_20", "ema_50", "atr", "rsi"]]).any():
        return StrategySignal("hold", None, None, None, 0.0, "Indicators not ready")
    # Pullback condition: price near EMA20 but above EMA50 and RSI neutral
    pullback = (
        last["close"] > last["ema_50"]
        and last["low"] <= last["pb_20"]
        and last["close"] > last["pb_20"]
        and 40 <= last["rsi"] <= 60
    )

    if not pullback:
        return StrategySignal("hold", None, None, None, 0.0, "No pullback detected for trend entry")

    entry = last["close"]
    stop = entry - 1.5 * last["atr"]
    target = entry + 3.0 * last["atr"]

    return StrategySignal(
        action="enter_long",
        entry=float(entry),
        stop_loss=float(stop),
        take_profit=float(target),
        confidence=0.55,
        reason="4h uptrend + 1h WMA pullback + RSI reset"
    )
