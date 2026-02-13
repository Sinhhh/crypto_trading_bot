import os
import sys
import logging
import argparse
import pandas as pd

from datetime import datetime

from utils.helpers import slice_lookback

# -------------------------------
# Project setup
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Custom modules
from strategies.smc_signal import generate_signal
from utils.data_loader import load_csv_data, normalize_ohlcv

# -------------------------------
# Logging setup
# -------------------------------
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)
log_filename = datetime.now().strftime("backtest_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_FOLDER, log_filename)

logger = logging.getLogger("backtest_smc")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# Console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(ch)
# File
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
POSITION_SIZE = 1000
LOOKBACK_15M = 50
LOOKBACK_1H = 200
LOOKBACK_4H = 100

# -------------------------------
# Helpers
# -------------------------------
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
    parser = argparse.ArgumentParser(description="SMC Backtest with Partial Exit + Trailing Stop")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list: BTC,ETH")
    parser.add_argument("--log-skips", action="store_true", help="Print per-candle SKIP reasons")
    return parser

# -------------------------------
# Trade simulation with Partial + Trailing
# -------------------------------
def simulate_trade_partial_trailing(df_15m, entry_idx, entry_price, stop, target, bias, max_bars=48, logger=None):
    exit_logs = []
    partial_taken = False
    remaining_position = 1.0  # 100%

    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df_15m))):
        candle = df_15m.iloc[j]
        price = candle["close"]

        # BUY logic
        if bias == "BUY":
            # Trailing stop after partial
            if partial_taken:
                new_stop = max(stop, price * 0.98)  # Trailing 2% stop
                stop = new_stop
            # Check stop
            if candle["low"] <= stop:
                pnl = (stop - entry_price) * POSITION_SIZE / entry_price * remaining_position
                exit_logs.append(("STOP", pnl, candle["timestamp"], remaining_position))
                return exit_logs
            # Partial exit at target
            if not partial_taken and candle["high"] >= target:
                pnl = (target - entry_price) * POSITION_SIZE / entry_price * 0.5
                partial_taken = True
                remaining_position = 0.5
                exit_logs.append(("PARTIAL_TARGET", pnl, candle["timestamp"], 0.5))

        # SELL logic
        elif bias == "SELL":
            if partial_taken:
                new_stop = min(stop, price * 1.02)  # Trailing up 2%
                stop = new_stop
            if candle["high"] >= stop:
                pnl = (entry_price - stop) * POSITION_SIZE / entry_price * remaining_position
                exit_logs.append(("STOP", pnl, candle["timestamp"], remaining_position))
                return exit_logs
            if not partial_taken and candle["low"] <= target:
                pnl = (entry_price - target) * POSITION_SIZE / entry_price * 0.5
                partial_taken = True
                remaining_position = 0.5
                exit_logs.append(("PARTIAL_TARGET", pnl, candle["timestamp"], 0.5))

    # Time exit if max bars reached
    last = df_15m.iloc[min(entry_idx + max_bars, len(df_15m)-1)]
    pnl = ((last["close"] - entry_price) if bias=="BUY" else (entry_price - last["close"])) * POSITION_SIZE / entry_price * remaining_position
    exit_logs.append(("TIME_EXIT", pnl, last["timestamp"], remaining_position))
    return exit_logs

# -------------------------------
# Backtest function
# -------------------------------
def backtest_symbol(symbol, log_skips=False):
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

        ts = df_15m.iloc[i]["timestamp"]
        current_15m = slice_lookback(df_15m, ts, LOOKBACK_15M)
        current_1h = slice_lookback(df_1h, ts, LOOKBACK_1H)
        current_4h = slice_lookback(df_4h, ts, LOOKBACK_4H)

        if len(current_4h) < 50 or current_1h.empty or current_4h.empty:
            if log_skips:
                logger.info(f"SKIP | {symbol} | ts={ts} | insufficient higher timeframe data")
            continue

        signals = generate_signal(current_4h, current_1h, current_15m)
        bias = signals["bias_4h"]
        setup = signals["setup_1h"]
        entry = signals["entry_15m"]

        if not setup["setup_valid"] or bias=="HOLD" or not entry["entry_signal"]:
            if log_skips:
                logger.info(f"SKIP | {symbol} | ts={ts} | bias={bias} | setup_valid={setup['setup_valid']} | entry_signal={entry['entry_signal']}")
            continue

        entry_price = entry["entry_price"]
        stop = entry["stop"]
        target = entry["target"]

        exit_logs = simulate_trade_partial_trailing(df_15m, i, entry_price, stop, target, bias, logger=logger)
        trade_exit_idx = i + len(exit_logs)
        in_trade = True

        for reason, pnl_usd, exit_ts, portion in exit_logs:
            capital += pnl_usd
            trades.append({
                "symbol": symbol,
                "entry_ts": ts,
                "exit_ts": exit_ts,
                "bias": bias,
                "entry": entry_price,
                "stop": stop,
                "target": target,
                "exit_reason": reason,
                "pnl_usd": pnl_usd,
                "capital": capital,
                "portion": portion,
            })
            logger.info(f"TRADE | {symbol} | {bias} | Entry: {entry_price:.2f} | Exit: {reason} | PnL: {pnl_usd:.2f} | Capital: {capital:.2f} | Portion: {portion*100:.0f}%")

    df_trades = pd.DataFrame(trades)
    total_pnl = df_trades["pnl_usd"].sum() if not df_trades.empty else 0.0
    winrate = (len(df_trades[df_trades["pnl_usd"]>0])/len(df_trades)*100) if len(df_trades) else 0.0
    print(f"\n=== BACKTEST SUMMARY | {symbol} ===")
    print(f"Trades: {len(df_trades)}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Final capital: {capital:.2f} USD")
    print(f"Total PnL: {total_pnl:.2f} USD")
    return df_trades

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        symbols = _parse_symbols(args.symbols)
    except ValueError as e:
        raise SystemExit(str(e))

    all_trades = []
    for sym in symbols:
        df_trades = backtest_symbol(sym, log_skips=args.log_skips)
        if df_trades is not None and not df_trades.empty:
            all_trades.append(df_trades)

    if all_trades:
        df_all = pd.concat(all_trades, ignore_index=True)
        summary_file = os.path.join(LOG_FOLDER, "all_trades_summary.csv")
        df_all.to_csv(summary_file, index=False)
        logger.info(f"Saved all trades summary to {summary_file}")