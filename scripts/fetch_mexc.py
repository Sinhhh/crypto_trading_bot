"""Fetch historical OHLCV from MEXC.

Module entrypoint:
  python3 -m scripts.fetch_mexc --timeframe 1h --symbols BTCUSDT,ETHUSDT

Outputs (default out-dir: data/raw/):
  data/raw/BTCUSDT_1H.csv
  data/raw/ETHUSDT_1H.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd

# Support running as a plain script: `python3 scripts/fetch_mexc.py ...`
# (Recommended usage remains: `python3 -m scripts.fetch_mexc ...`)
if __package__ is None:  # pragma: no cover
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from crypto_trading.utils.timeframes import timeframe_to_suffix


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}/{quote}"
    if s.endswith("USDT") and len(s) > 4:
        base = s[: -4]
        return f"{base}/USDT"
    return s


def _symbol_to_filename(symbol_ccxt: str) -> str:
    return symbol_ccxt.replace("/", "").upper()


def _timeframe_to_suffix(timeframe: str) -> str:
    return timeframe_to_suffix(timeframe)


def fetch_all_ohlcv(
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    save_path: str,
    limit: int = 1000,
) -> pd.DataFrame:
    exchange = ccxt.mexc(
        {
            "timeout": 30000,
            "enableRateLimit": True,
        }
    )

    symbol_ccxt = _normalize_symbol(symbol)
    tf = str(timeframe).strip().lower()
    since = exchange.parse8601(str(start_date))
    all_ohlcv: list[list] = []

    print(f"Fetching {symbol_ccxt} {tf} candles from {start_date}...")

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol_ccxt, tf, since=since, limit=int(limit))
        if not ohlcv:
            break

        all_ohlcv += ohlcv
        last_timestamp = ohlcv[-1][0]
        since = int(last_timestamp) + 1

        print(
            f"Fetched {len(all_ohlcv)} candles; last candle: {datetime.utcfromtimestamp(last_timestamp / 1000)}"
        )

        now_ts = exchange.milliseconds()
        if since >= now_ts:
            break

    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    if df.empty:
        raise RuntimeError(
            f"No OHLCV candles returned for {symbol_ccxt} {tf} since {start_date}. "
            "Refusing to write an empty (header-only) CSV."
        )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(
        drop=True
    )

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} candles to {out_path}")
    return df


def fetch_all_symbols(
    *,
    symbols: list[str],
    timeframe: str,
    start_date: str,
    out_dir: str = "data/raw",
) -> None:
    for sym in symbols:
        symbol_ccxt = _normalize_symbol(sym)
        name = _symbol_to_filename(symbol_ccxt)
        tf_suffix = _timeframe_to_suffix(timeframe)
        save_path = str(Path(out_dir) / f"{name}_{tf_suffix}.csv")
        fetch_all_ohlcv(
            symbol=symbol_ccxt,
            timeframe=timeframe,
            start_date=start_date,
            save_path=save_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch MEXC historical OHLCV for symbols/timeframe"
    )
    parser.add_argument("--timeframe", default="1h", help="Timeframe, e.g. 15m, 1h, 4h")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT or BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--start-date",
        default="2025-01-01T00:00:00Z",
        help="ISO8601 UTC, e.g. 2025-01-01T00:00:00Z",
    )
    parser.add_argument("--out-dir", default="data/raw", help="Output directory for CSVs")
    args = parser.parse_args()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    fetch_all_symbols(
        symbols=symbols,
        timeframe=str(args.timeframe),
        start_date=str(args.start_date),
        out_dir=str(args.out_dir),
    )


if __name__ == "__main__":
    main()
