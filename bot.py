import os

import ccxt
import pandas as pd

from indicators.bb_atr import atr
from regimes.regime_detector import detect_regime
from strategies.breakout import breakout_strategy
from strategies.mean_reversion import mean_reversion_strategy
from strategies.trend_participation import (
    detect_trend_permission,
    trend_participation_strategy,
)

exchange_config: dict = {"enableRateLimit": True}

# Optional (only needed for private endpoints / live trading).
# Set env vars instead of hardcoding secrets in source code.
api_key = os.getenv("MEXC_API_KEY")
api_secret = os.getenv("MEXC_API_SECRET")
if api_key and api_secret:
    exchange_config.update({"apiKey": api_key, "secret": api_secret})

exchange = ccxt.mexc(exchange_config)

# Sync time difference between local and exchange server
exchange.load_time_difference()


# -----------------------------
# 2️⃣ Fetch OHLC function
# -----------------------------
def fetch_ohlc(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLC data from MEXC via CCXT"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def volume_confirmation(df: pd.DataFrame, multiplier=1.5) -> bool:
    """Returns True if latest volume is > multiplier * 20-period avg."""
    if len(df) < 20:
        return False
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    latest_vol = df["volume"].iloc[-1]
    if pd.isna(avg_vol) or pd.isna(latest_vol):
        return False
    return latest_vol > avg_vol * multiplier

def stochastic_filter(df: pd.DataFrame, k_period=14, d_period=3) -> float:
    """Return %K of Stochastic oscillator for latest candle (0-100)."""
    if len(df) < k_period:
        return 50  # Neutral if not enough data
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min)
    if pd.isna(denom.iloc[-1]) or float(denom.iloc[-1]) == 0.0:
        return 50
    percent_k = 100 * (df["close"] - low_min) / denom
    percent_d = percent_k.rolling(d_period).mean()
    latest = percent_d.iloc[-1]
    if pd.isna(latest):
        return 50
    return float(latest)

def opening_range_breakout(df: pd.DataFrame, k=0.5) -> bool:
    """Return True if latest close exceeds today's opening range + k*ATR.

    Uses the first candle of the latest day in `df` as the "opening range".
    """
    if len(df) < 2 or "timestamp" not in df.columns:
        return False
    latest_day = df["timestamp"].iloc[-1].date()
    df_day = df[df["timestamp"].dt.date == latest_day]
    if len(df_day) < 2:
        return False

    first_candle = df_day.iloc[0]
    range_val = first_candle["high"] - first_candle["low"]
    latest_close = df["close"].iloc[-1]
    atr_val = atr(df).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return False
    return latest_close > first_candle["high"] + k * atr_val


# -----------------------------
# 3️⃣ Main Signal Generator
# -----------------------------
def generate_signal(symbol: str):
    # Fetch 1h & 4h data
    df_1h = fetch_ohlc(symbol, "1h", limit=500)
    df_4h = fetch_ohlc(symbol, "4h", limit=500)

    # --- 3a. Detect market regime ---
    regime = detect_regime(df_4h)
    print(f"Market regime detected: {regime}")

    # --- 3b. Select strategy based on regime ---
    source_strategy = "breakout"

    if regime in ["TREND_UP", "TREND_DOWN"]:
        # Trend participation first
        trend_dir = detect_trend_permission(df_4h)
        signal = trend_participation_strategy(df_1h, trend_dir)
        source_strategy = "trend_participation"
        if signal.action == "hold":
            # Fallback to breakout if no trend entry signal
            signal = breakout_strategy(df_1h)
            source_strategy = "breakout"
    elif regime == "RANGE":
        signal = mean_reversion_strategy(df_1h)
        source_strategy = "mean_reversion"
    else:  # TRANSITION
        signal = breakout_strategy(df_1h)  # placeholder strategy
        source_strategy = "breakout"

    if signal.action == "enter_long":
        # Volume filter
        if not volume_confirmation(df_1h):
            signal.action = "hold"
            signal.reason += " | SKIP: low volume"

        # Stochastic filter for mean reversion entries
        if source_strategy == "mean_reversion":
            stoch = stochastic_filter(df_1h)
            if stoch > 80:  # overbought
                signal.action = "hold"
                signal.reason += f" | SKIP: stochastic overbought ({stoch:.1f})"

        # ORB filter for breakouts
        if source_strategy == "breakout":
            if not opening_range_breakout(df_1h):
                signal.action = "hold"
                signal.reason += " | SKIP: no ORB"

    return signal


# -----------------------------
# 4️⃣ Entry Point
# -----------------------------
if __name__ == "__main__":
    SYMBOL = "BTC/USDT"  # Example
    signal = generate_signal(SYMBOL)
    print(f"Generated signal for {SYMBOL}: {signal}")
