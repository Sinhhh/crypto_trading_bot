import pandas as pd

from dataclasses import dataclass

from indicators.bb_atr import atr


@dataclass
class StrategySignal:
    action: str                   # "enter_long" or "hold"
    entry: float | None           # Entry price if applicable
    stop_loss: float | None       # Stop-loss level
    take_profit: float | None     # Take-profit level
    confidence: float             # Heuristic confidence [0,1]
    reason: str                   # Human-readable rationale


def breakout_strategy(df_1h: pd.DataFrame, period: int = 3) -> StrategySignal:
    """
    Breakout/Momentum strategy for trending markets.

    Entry: price closes above the high of the last `period` candles
    Stop Loss: 1 ATR below entry
    Take Profit: 2 ATR above entry

    Returns a `StrategySignal` with action "enter_long" or "hold".
    """
    if len(df_1h) < period + 1:
        return StrategySignal(
            action="hold",
            entry=None,
            stop_loss=None,
            take_profit=None,
            confidence=0.0,
            reason="Insufficient data for breakout"
        )

    df = df_1h.copy()
    df["atr"] = atr(df)

    latest = df.iloc[-1]
    recent_high = df["high"].iloc[-(period + 1):-1].max()  # last N candles excluding current

    # --- Check ATR readiness ---
    if pd.isna(latest["atr"]):
        return StrategySignal(
            action="hold",
            entry=None,
            stop_loss=None,
            take_profit=None,
            confidence=0.0,
            reason="ATR not ready"
        )

    # Breakout condition
    if latest["close"] > recent_high:
        entry = latest["close"]
        stop_loss = float(entry - latest["atr"])       # 1 ATR below entry
        take_profit = float(entry + 2 * latest["atr"]) # 2 ATR target

        # Safety checks
        if stop_loss >= entry:
            return StrategySignal(
                action="hold",
                entry=None,
                stop_loss=None,
                take_profit=None,
                confidence=0.0,
                reason="Invalid stop-loss (>= entry)"
            )
        if take_profit <= entry:
            return StrategySignal(
                action="hold",
                entry=None,
                stop_loss=None,
                take_profit=None,
                confidence=0.0,
                reason="Invalid take-profit (<= entry)"
            )

        confidence = 0.6
        reason = f"Price closed above last {period} highs ({recent_high:.2f})"

        return StrategySignal(
            action="enter_long",
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason
        )

    return StrategySignal(
        action="hold",
        entry=None,
        stop_loss=None,
        take_profit=None,
        confidence=0.0,
        reason=f"No breakout above last {period} highs"
    )
