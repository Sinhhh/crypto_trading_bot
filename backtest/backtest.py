import os
import sys
import logging
import argparse

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
logger = logging.getLogger("backtest_smc")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# -------------------------------
# Config
# -------------------------------
DEFAULT_SYMBOLS = ["BTC", "ETH"]
INITIAL_CAPITAL = 10000
POSITION_SIZE = 1000  # USD per trade

# Logging verbosity: when False, suppress per-candle SKIP logs.
LOG_SKIPS = False

LOOKBACK_15M = 20
LOOKBACK_1H = 120
LOOKBACK_4H = 60

TRADE_LOG_CSV = "backtest_smc_trades.csv"


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SYMBOLS)
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    allowed = {"BTC", "ETH"}
    invalid = [s for s in symbols if s not in allowed]
    if invalid:
        raise ValueError(f"Invalid symbols: {invalid}. Allowed: BTC, ETH")
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMC multi-timeframe backtest (BTC/ETH)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list: BTC,ETH (default: BTC,ETH)",
    )
    parser.add_argument(
        "--log-skips",
        action="store_true",
        help="Print per-candle SKIP reasons (very verbose)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=TRADE_LOG_CSV,
        help=f"Output CSV path (default: {TRADE_LOG_CSV})",
    )
    return parser


# -------------------------------
# Helper: slice lookbacks
# -------------------------------
def slice_lookback(df, current_ts, lookback):
    df_slice = df[df["timestamp"] <= current_ts].tail(lookback)
    return df_slice


# -------------------------------
# Backtest per symbol (safe version)
# -------------------------------
def backtest_symbol_safe(symbol):
    capital = INITIAL_CAPITAL
    trades = []

    # Load & normalize data
    df_4h = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_4H.csv"))
    df_1h = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_1H.csv"))
    df_15m = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_15M.csv"))

    print(f"=== Starting backtest for {symbol} | Capital: {capital} ===\n")

    for i in range(2, len(df_15m)):
        # Safe slice for 15M lookback
        start_idx_15m = max(0, i - LOOKBACK_15M)
        current_15m = df_15m.iloc[start_idx_15m : i + 1]
        if current_15m.empty:
            continue  # skip if no data

        ts = current_15m.iloc[-1]["timestamp"]

        # Safe slices for higher timeframes
        current_1h = slice_lookback(df_1h, ts, LOOKBACK_1H)
        current_4h = slice_lookback(df_4h, ts, LOOKBACK_4H)

        if current_1h.empty or current_4h.empty:
            if LOG_SKIPS:
                logger.info(
                    f"SKIP | {symbol} | Reason: EMPTY_HIGHER_TIMEFRAME | Timestamp: {ts}"
                )
            continue

        # Generate multi-timeframe signal
        signals = generate_signal(current_4h, current_1h, current_15m)
        bias = signals["bias_4h"]
        setup = signals["setup_1h"]
        entry = signals["entry_15m"]

        # Skip conditions
        if not setup["setup_valid"] or bias == "HOLD" or not entry["entry_signal"]:
            if LOG_SKIPS:
                logger.info(
                    f"SKIP | {symbol} | Reason: {entry['reason']} | Bias: {bias} | Entry Candle Close: {current_15m.iloc[-1]['close']:.2f}"
                )
            continue

        # Simulate trade
        entry_price = entry["entry_price"]
        stop = entry["stop"]
        target = entry["target"]

        if bias == "BUY":
            pnl = (target - entry_price) * (POSITION_SIZE / entry_price)
        else:
            pnl = (entry_price - target) * (POSITION_SIZE / entry_price)

        capital += pnl

        trades.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "bias": bias,
                "setup_valid": setup["setup_valid"],
                "entry": entry_price,
                "stop": stop,
                "target": target,
                "pnl_usd": pnl,
                "capital": capital,
            }
        )

        logger.info(
            f"TRADE | {symbol} | {bias} | Entry: {entry_price:.2f} | Stop: {stop:.2f} | Target: {target:.2f} | "
            f"PnL: {pnl:.2f} | Capital: {capital:.2f}"
        )

    df_trades = pd.DataFrame(trades)

    # Summary
    total_pnl = df_trades["pnl_usd"].sum() if not df_trades.empty else 0.0
    win_trades = (
        df_trades[df_trades["pnl_usd"] > 0] if not df_trades.empty else df_trades
    )
    winrate = (len(win_trades) / len(df_trades) * 100) if len(df_trades) else 0.0

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Symbol: {symbol}")
    print(f"Trades: {len(df_trades)}")
    if not df_trades.empty:
        buy_count = int((df_trades["bias"] == "BUY").sum())
        sell_count = int((df_trades["bias"] == "SELL").sum())
    else:
        buy_count = 0
        sell_count = 0
    print(f"BUY trades: {buy_count} | SELL trades: {sell_count}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Final capital: {capital:.2f} USD")
    print(f"Total PnL: {total_pnl:.2f} USD")

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

    trade_log_csv = args.out

    all_trades = []
    for sym in symbols:
        df = backtest_symbol_safe(sym)
        if df is not None and not df.empty:
            all_trades.append(df)

    if all_trades:
        df_all = pd.concat(all_trades, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    df_all.to_csv(trade_log_csv, index=False)
    print(f"Backtest completed. Trade log saved to: {trade_log_csv}")
