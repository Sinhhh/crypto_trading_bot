"""
Fetch historical OHLCV from MEXC for one or more symbols and a chosen timeframe.

Examples:
    python3 scripts/mexc_fetch_ohlcv.py --timeframe 15m
    python3 scripts/mexc_fetch_ohlcv.py --timeframe 1h --symbols BTCUSDT,ETHUSDT --start-date 2025-01-01T00:00:00Z

Outputs (default out-dir: data/):
    data/raw/BTCUSDT_15M.csv
    data/raw/ETHUSDT_15M.csv

CSV columns: datetime, open, high, low, close, volume
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd


if __package__ is None:  # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    exchange = ccxt.mexc({
        "timeout": 30000,
    })
    symbol_ccxt = _normalize_symbol(symbol)
    since = exchange.parse8601(start_date)
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol_ccxt,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break