import os
import sys
import logging
import argparse
from datetime import datetime

import pandas as pd


# Allow running this file directly (python backtest/backtest.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.strategies.multi_timeframe import generate_signal
from src.utils.data_loader import load_csv_data, normalize_ohlcv

# -------------------------------
# Logging setup
# -------------------------------
LOG_FOLDER = os.path.join("backtest", "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)

log_filename = datetime.now().strftime("backtest_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_FOLDER, log_filename)

logger = logging.getLogger("backtest_smc")
logger.setLevel(logging.INFO)

# Clear previous handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(log_path, mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(fh)

logger.info(f"Backtest started, log file: {log_path}")

# -------------------------------
# Config
# -------------------------------
DEFAULT_SYMBOLS = ["BTC"]
INITIAL_CAPITAL = 10000
POSITION_SIZE = 1000  # USD per trade

LOG_SKIPS = False
LOOKBACK_15M = 50  # 12–13h
LOOKBACK_1H = 200  # ~8+ days
LOOKBACK_4H = 100  # ~16+ days



def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SYMBOLS)
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    allowed = {"BTC", "ETH"}
    invalid = [s for s in symbols if s not in allowed]
    if invalid:
        raise ValueError(f"Invalid symbols: {invalid}. Allowed: BTC, ETH")
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SMC multi-timeframe backtest (BTC/ETH)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list: BTC,ETH")
    parser.add_argument("--log-skips", action="store_true", help="Print per-candle SKIP reasons")
    return parser


# -------------------------------
# Helper: slice lookbacks
# -------------------------------
def slice_lookback(df, current_ts, lookback):
    return df[df["timestamp"] <= current_ts].tail(lookback)


def simulate_trade_15m(
    df_15m: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop: float,
    target: float,
    bias: str,
    max_bars: int = 48,
):
    """
    Simulate forward candles after entry.
    Returns: pnl_price, exit_idx, exit_ts, exit_reason
    """
    end_idx = min(entry_idx + max_bars, len(df_15m) - 1)

    for j in range(entry_idx + 1, end_idx + 1):
        candle = df_15m.iloc[j]

        if bias == "BUY":
            if candle["low"] <= stop:
                return stop - entry_price, j, candle["timestamp"], "STOP"
            if candle["high"] >= target:
                return target - entry_price, j, candle["timestamp"], "TARGET"

        elif bias == "SELL":
            if candle["high"] >= stop:
                return entry_price - stop, j, candle["timestamp"], "STOP"
            if candle["low"] <= target:
                return entry_price - target, j, candle["timestamp"], "TARGET"

    last = df_15m.iloc[end_idx]
    pnl = (last["close"] - entry_price) if bias == "BUY" else (entry_price - last["close"])
    return pnl, end_idx, last["timestamp"], "TIME_EXIT"


# -------------------------------
# Backtest per symbol
# -------------------------------
def backtest_symbol_safe(symbol):
    capital = INITIAL_CAPITAL
    trades = []

    df_4h = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_4H.csv"))
    df_1h = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_1H.csv"))
    df_15m = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_15M.csv"))

    logger.info(f"=== Starting backtest for {symbol} | Capital: {capital} ===\n")

    in_trade = False
    trade_exit_idx = None

    for i in range(2, len(df_15m)):
        if in_trade and i <= trade_exit_idx:
            continue
        elif in_trade:
            in_trade = False
            trade_exit_idx = None

        current_15m = df_15m.iloc[max(0, i - LOOKBACK_15M) : i + 1]
        if current_15m.empty:
            continue
        ts = current_15m.iloc[-1]["timestamp"]

        current_1h = slice_lookback(df_1h, ts, LOOKBACK_1H)
        current_4h = slice_lookback(df_4h, ts, LOOKBACK_4H)

        if len(current_4h) < 50:  # require at least 50 candles
            if LOG_SKIPS:
                logger.info(f"SKIP | {symbol} | ts={ts} | NOT_ENOUGH_4H_CANDLES")
            continue

        if current_1h.empty or current_4h.empty:
            if LOG_SKIPS:
                logger.info(f"SKIP | {symbol} | ts={ts} | EMPTY_HIGHER_TIMEFRAME")
            continue

        signals = generate_signal(current_4h, current_1h, current_15m)
        bias = signals["bias_4h"]
        setup = signals["setup_1h"]
        entry = signals["entry_15m"]

        if not setup["setup_valid"] or bias == "HOLD" or not entry["entry_signal"]:
            if LOG_SKIPS:
                logger.info(f"SKIP | {symbol} | bias={bias} | setup_valid={setup['setup_valid']} | reason={entry.get('reason','N/A')}")
            continue

        entry_price = entry["entry_price"]
        stop = entry["stop"]
        target = entry["target"]

        pnl_price, exit_idx, exit_ts, exit_reason = simulate_trade_15m(
            df_15m, i, entry_price, stop, target, bias
        )

        pnl_usd = pnl_price * (POSITION_SIZE / entry_price)
        capital += pnl_usd

        in_trade = True
        trade_exit_idx = exit_idx

        trades.append({
            "symbol": symbol,
            "entry_ts": ts,
            "exit_ts": exit_ts,
            "bias": bias,
            "entry": entry_price,
            "stop": stop,
            "target": target,
            "exit_reason": exit_reason,
            "pnl_usd": pnl_usd,
            "capital": capital,
        })

        logger.info(f"TRADE | {symbol} | {bias} | Entry: {entry_price:.2f} | Exit: {exit_reason} | PnL: {pnl_usd:.2f} | Capital: {capital:.2f}")

    df_trades = pd.DataFrame(trades)

    # Summary
    total_pnl = df_trades["pnl_usd"].sum() if not df_trades.empty else 0.0
    winrate = (len(df_trades[df_trades["pnl_usd"] > 0]) / len(df_trades) * 100) if len(df_trades) else 0.0
    buy_count = int((df_trades["bias"] == "BUY").sum()) if not df_trades.empty else 0
    sell_count = int((df_trades["bias"] == "SELL").sum()) if not df_trades.empty else 0

    logger.info(f"\n=== BACKTEST SUMMARY | {symbol} ===")
    logger.info(f"Trades: {len(df_trades)} | BUY: {buy_count} | SELL: {sell_count}")
    logger.info(f"Winrate: {winrate:.2f}% | Final capital: {capital:.2f} USD | Total PnL: {total_pnl:.2f} USD")

    return df_trades


# -------------------------------
# Run safe backtest
# -------------------------------
if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        symbols = _parse_symbols(args.symbols)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.log_skips:
        LOG_SKIPS = True

    all_trades = []
    for sym in symbols:
        df = backtest_symbol_safe(sym)
        if df is not None and not df.empty:
            all_trades.append(df)

    df_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
