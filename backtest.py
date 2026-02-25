#!/usr/bin/env python3
# backtest.py — CSV backtest runner using RiskManagerV2 + PaperTrader

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# -------------------------------
# Project setup
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# -------------------------------
# Custom modules
# -------------------------------
from utils.helpers import slice_lookback
from data.data_loader import load_csv_data, normalize_ohlcv
from core.strategies.smc_signal import generate_signal
from core.indicators.atr import compute_atr
from core.risk.risk_manager import RiskManagerV2
from core.risk.risk_config import RiskConfigV2
from core.trader.paper_trader import PaperTrader
from core.broker.broker_sim import BrokerSim

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
INITIAL_CAPITAL = 10_000
LOOKBACK_15M = 50
LOOKBACK_1H = 200
LOOKBACK_4H = 100

# RiskManagerV2 config
RISK_CFG = RiskConfigV2(
    base_risk_pct=0.01,
    min_risk_pct=0.0025,
    max_portfolio_risk_pct=0.05,
    dd_soft_limit=0.05,
    dd_hard_limit=0.10,
    max_symbol_risk_pct=0.02,
    min_rr=2.0,
    enable_quality_scaling=True,
)


# -------------------------------
# CLI parser
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
    parser = argparse.ArgumentParser(
        description="SMC Backtest with RiskManagerV2 + ATR TP/SL"
    )
    parser.add_argument(
        "--symbols", type=str, default=None, help="Comma-separated list: BTC,ETH"
    )
    parser.add_argument(
        "--log-skips", action="store_true", help="Print per-candle SKIP reasons"
    )
    return parser


# -------------------------------
# Trade simulation
# -------------------------------
def simulate_trade_dynamic_atr(trader: PaperTrader, df_15m, entry_idx, symbol, bias):
    """
    Simulate trade exit based on ATR trailing stops and targets
    """
    atr_series = compute_atr(df_15m, period=14)
    entry_candle = df_15m.iloc[entry_idx]
    entry_price = entry_candle["close"]

    for j in range(entry_idx + 1, len(df_15m)):
        candle = df_15m.iloc[j]
        price = candle["close"]
        atr = atr_series.iloc[j] if j < len(atr_series) else atr_series.iloc[-1]

        # Update broker price
        trader.broker.set_last_price(symbol, price)

        # Check stop / target
        exit_data = trader.broker.check_stop_target(symbol)
        if exit_data:
            yield exit_data


# -------------------------------
# Backtest loop
# -------------------------------
def backtest_symbol(symbol: str, log_skips=False):
    # Setup broker and trader
    broker = BrokerSim(starting_cash_usdt=INITIAL_CAPITAL)
    risk_manager = RiskManagerV2(broker, RISK_CFG)
    trader = PaperTrader(
        cfg=None, exchange=None, broker=broker, risk=risk_manager, logger=logger
    )

    trades_log = []

    df_4h = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_4H.csv"))
    df_1h = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_1H.csv"))
    df_15m = normalize_ohlcv(load_csv_data(f"data/raw/{symbol}_15M.csv"))

    print(f"=== Starting backtest for {symbol} | Capital: {INITIAL_CAPITAL} ===\n")

    for i in range(2, len(df_15m)):
        ts = df_15m.iloc[i]["timestamp"]
        current_15m = slice_lookback(df_15m, ts, LOOKBACK_15M)
        current_1h = slice_lookback(df_1h, ts, LOOKBACK_1H)
        current_4h = slice_lookback(df_4h, ts, LOOKBACK_4H)

        if len(current_4h) < 50 or current_1h.empty or current_4h.empty:
            if log_skips:
                logger.info(
                    f"SKIP | {symbol} | ts={ts} | insufficient higher timeframe data"
                )
            continue

        signals = generate_signal(current_4h, current_1h, current_15m)
        bias = signals.get("bias_4h", "HOLD")
        setup = signals.get("setup_1h", {})
        entry = signals.get("entry_15m", {})

        if (
            bias == "HOLD"
            or not setup.get("setup_valid")
            or not entry.get("entry_signal")
        ):
            if log_skips:
                logger.info(f"SKIP | {symbol} | ts={ts} | reason=signal invalid")
            continue

        entry_price = entry.get("entry_price")
        stop_price = entry.get(
            "stop_price", entry_price * 0.99 if bias == "BUY" else entry_price * 1.01
        )
        target_price = entry.get(
            "target_price", entry_price * 1.02 if bias == "BUY" else entry_price * 0.98
        )

        # Compute size
        size = risk_manager.compute_size(
            symbol, bias, entry_price, stop_price, target_price
        )
        if size <= 0:
            if log_skips:
                logger.info(f"SKIP | {symbol} | ts={ts} | reason=risk sizing zero")
            continue

        # Open position
        if bias == "BUY":
            resp = broker.open_long(symbol, size, entry_price, stop_price, target_price)
        else:
            resp = broker.open_short(
                symbol, size, entry_price, stop_price, target_price
            )

        if not resp.get("ok"):
            if log_skips:
                logger.info(f"SKIP | {symbol} | ts={ts} | reason={resp.get('reason')}")
            continue

        # Track exit via ATR
        for exit_data in simulate_trade_dynamic_atr(trader, df_15m, i, symbol, bias):
            trades_log.append(exit_data)

    # -------------------------
    # Summary
    # -------------------------
    total_pnl = sum(d["pnl"] for d in trades_log) if trades_log else 0.0
    winrate = (
        (len([d for d in trades_log if d["pnl"] > 0]) / len(trades_log) * 100)
        if trades_log
        else 0.0
    )

    print(f"\n=== BACKTEST SUMMARY | {symbol} ===")
    print(f"Trades: {len(trades_log)}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Final capital: {broker.equity_usdt():.2f} USD")
    print(f"Total PnL: {total_pnl:.2f} USD")

    return pd.DataFrame(trades_log)


# -------------------------------
# Main entrypoint
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
