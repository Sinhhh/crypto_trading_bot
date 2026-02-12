"""
Optimized candle fetcher for BTC & ETH
- Multi-timeframe (15M / 1H / 4H)
- Ensures enough candles for SMC framework:
    * liquidity grab
    * displacement
    * micro BOS
    * ATR, OB/FVG, swing high/low
- Saves CSV for backtest
"""

from datetime import datetime
import os
import argparse
from utils.market_data import fetch_ohlcv
from utils.data_loader import normalize_ohlcv

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/raw/"
SYMBOLS = ["BTC", "ETH"]
TIMEFRAMES = ["15M", "1H", "4H"]

DEFAULT_HISTORY_DAYS = 90

# Lookbacks for SMC framework (enough candles for ATR, BOS, FVG, liquidity)
LOOKBACKS = {
    "15M": 30,  # liquidity + micro BOS
    "1H": 60,  # setup validation
    "4H": 100,  # bias + supply/demand
}


# -------------------------------
# UTILITIES
# -------------------------------
def ensure_data_path():
    os.makedirs(DATA_PATH, exist_ok=True)


def parse_since(value: str | None) -> int | None:
    """Return unix ms timestamp for --since input"""
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError("Invalid --since format. Use YYYY-MM-DD or unix ms timestamp.")


def calculate_limit(timeframe: str, history_days: int) -> int:
    """
    Calculate total candles to fetch based on:
    - history_days
    - timeframe
    - lookback for SMC framework
    """
    if timeframe.endswith("M"):
        minutes = int(timeframe[:-1])
        candles_per_day = (24 * 60) // minutes
    elif timeframe.endswith("H"):
        hours = int(timeframe[:-1])
        candles_per_day = 24 // hours
    else:
        raise ValueError(f"Unsupported timeframe {timeframe}")

    lookback = LOOKBACKS.get(timeframe, 50)
    limit = candles_per_day * history_days + lookback * 2  # extra safety
    return limit


def fetch_and_save(symbol: str, timeframe: str, limit: int, since: int | None):
    print(f"[FETCH] {symbol} {timeframe} | limit={limit} | since={since}")
    df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, since=since)
    df = normalize_ohlcv(df)
    file_path = f"{DATA_PATH}{symbol}_{timeframe}.csv"
    df.to_csv(file_path, index=False)
    print(f"[SAVED] {len(df)} candles → {file_path}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch candles for BTC/ETH backtest")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Start date: YYYY-MM-DD or unix timestamp (ms)",
    )
    parser.add_argument(
        "--history_days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Number of days of history to fetch",
    )
    args = parser.parse_args()

    since = parse_since(args.since)
    ensure_data_path()

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            limit = calculate_limit(tf, args.history_days)
            fetch_and_save(symbol, tf, limit, since)


if __name__ == "__main__":
    main()
