import pandas as pd


def breakout_signal(
    df_1h: pd.DataFrame,
    period: int = 3,
) -> str:
    """Breakout (momentum) signal.

    Returns only: "BUY" / "SELL" / "HOLD".

    BUY: close breaks above recent highs.
    SELL: close breaks below recent lows (mirror).
    """
    if df_1h is None or len(df_1h) < int(period) + 1:
        return "HOLD"

    if not {"high", "low", "close"}.issubset(df_1h.columns):
        return "HOLD"

    df = df_1h
    latest = df.iloc[-1]
    recent_high = float(df["high"].iloc[-(period + 1) : -1].max())
    recent_low = float(df["low"].iloc[-(period + 1) : -1].min())
    close = float(latest["close"])

    if close > recent_high:
        return "BUY"
    if close < recent_low:
        return "SELL"
    return "HOLD"
