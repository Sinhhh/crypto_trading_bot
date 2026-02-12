import argparse
import logging
import os
import sys
import time
import math

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from strategies.multi_timeframe import generate_signal
from utils.data_loader import normalize_ohlcv

from exchange.ccxt_exchange import CCXTExchange
from broker.broker_sim import BrokerSim
from broker.risk_manager import RiskConfig, RiskManager


@dataclass(frozen=True)
class PaperConfig:
    exchange_id: str = "binance"
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    tf_4h: str = "4h"
    tf_1h: str = "1h"
    tf_15m: str = "15m"
    limit_4h: int = 120
    limit_1h: int = 240
    limit_15m: int = 240
    poll_seconds: int = 30
    close_on_sell_bias: bool = True
    log_path: str = "logs/paper.log"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("paper_trader")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    base, ext = os.path.splitext(log_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_log_path = f"{base}_{ts}{ext or '.log'}"
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(run_log_path, encoding="utf-8", mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Paper log file: {run_log_path}")
    return logger



class PaperTrader:
    def __init__(
        self,
        cfg: PaperConfig,
        exchange: CCXTExchange,
        broker: BrokerSim,
        risk: RiskManager,
        logger: logging.Logger,
    ):
        self.cfg = cfg
        self.ex = exchange
        self.broker = broker
        self.risk = risk
        self.logger = logger

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        return str(symbol).split("/")[0].upper()

    @staticmethod
    def _fmt_px(px) -> str:
        try:
            if px is None:
                return "NA"
            px_f = float(px)
            if math.isnan(px_f) or math.isinf(px_f):
                return "NA"
            return f"{px_f:.2f}"
        except Exception:
            return "NA"

    def _log_line(
        self,
        event: str,
        base_symbol: str,
        side: str,
        entry: float | None,
        pnl: float,
        capital: float,
        reason: str | None = None,
    ) -> None:
        exit_reason = reason or "OPEN"
        msg = (
            f"{event} | {base_symbol} | {side} | "
            f"Entry: {self._fmt_px(entry)} | Exit: {exit_reason} | "
            f"PnL: {float(pnl):.2f} | Capital: {float(capital):.2f}"
        )
        self.logger.info(msg)

    def _fetch_context(
        self, symbol: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_4h = self.ex.fetch_ohlcv_df(symbol, self.cfg.tf_4h, self.cfg.limit_4h)
        df_1h = self.ex.fetch_ohlcv_df(symbol, self.cfg.tf_1h, self.cfg.limit_1h)
        df_15m = self.ex.fetch_ohlcv_df(symbol, self.cfg.tf_15m, self.cfg.limit_15m)
        return normalize_ohlcv(df_4h), normalize_ohlcv(df_1h), normalize_ohlcv(df_15m)

    def _log_event(
        self,
        symbol: str,
        event: str,
        bias: str | None,
        setup_valid: bool | None,
        entry_signal: bool | None,
        entry: float | None,
        stop: float | None,
        target: float | None,
        amount: float | None,
        filled_price: float | None,
        pnl_usd: float | None,
        equity_usdt: float | None,
        reason: str | None,
    ) -> None:
        if event == "ERROR":
            self.logger.error(
                "ERROR | %s | reason=%s",
                symbol,
                reason or "UNKNOWN_ERROR",
            )
            return

        self.logger.info(
            "SKIP | %s | bias=%s | setup_valid=%s | entry_signal=%s | reason=%s",
            symbol,
            bias,
            setup_valid,
            entry_signal,
            reason or event,
        )

    def step(self, symbol: str) -> None:
        last_price = self.ex.fetch_last_price(symbol)
        if last_price is None:
            return
        self.broker.set_last_price(symbol, last_price)

        base_symbol = self._base_symbol(symbol)
        pos_before = self.broker.get_position(symbol)
        exit_fill = self.broker.check_stop_target(symbol)
        if exit_fill:
            eq_after = self.broker.equity_usdt()
            if pos_before and pos_before.side == "LONG":
                exit_side = "SELL"
            elif pos_before and pos_before.side == "SHORT":
                exit_side = "BUY"
            else:
                exit_side = "EXIT"

            entry_val = float(pos_before.entry) if pos_before else None
            stop_val = float(pos_before.stop) if pos_before else None
            target_val = float(pos_before.target) if pos_before else None
            amount_val = float(pos_before.amount) if pos_before else None

            self._log_line(
                event="TRADE",
                base_symbol=base_symbol,
                side=exit_side,
                entry=entry_val,
                pnl=float(exit_fill["pnl"]),
                capital=eq_after,
                reason=str(exit_fill.get("reason")),
            )
            self._log_event(
                symbol, f"EXIT_{exit_fill['reason']}", None, None, None,
                entry_val, stop_val, target_val, amount_val,
                exit_fill.get("filled_price"), float(exit_fill["pnl"]),
                eq_after, exit_fill.get("reason")
            )
        df_4h, df_1h, df_15m = self._fetch_context(symbol)
        sig = generate_signal(df_4h, df_1h, df_15m)

        bias = sig["bias"]
        setup_valid = bool(sig["setup_valid"])
        entry_signal = bool(sig["entry_signal"])

        entry_meta = sig.get("entry_15m") or {}
        entry_price = (
            entry_meta.get("entry_price")
            or entry_meta.get("filled_price")
        )
        stop = entry_meta.get("stop")
        target = entry_meta.get("target")
        entry_reason = entry_meta.get("reason")
        equity = self.broker.equity_usdt()

        pos_now = self.broker.get_position(symbol)
        if self.cfg.close_on_sell_bias and pos_now and bias in ("BUY", "SELL"):
            if pos_now.side == "LONG" and bias == "SELL":
                fill = self.broker.close_long(symbol, float(last_price), reason="BIAS_FLIP")
            elif pos_now.side == "SHORT" and bias == "BUY":
                fill = self.broker.close_short(symbol, float(last_price), reason="BIAS_FLIP")
            else:
                fill = None

            if fill and fill.get("ok"):
                eq_after = self.broker.equity_usdt()
                exit_side = "SELL" if pos_now.side == "LONG" else "BUY"
                self._log_line(
                    event="TRADE",
                    base_symbol=base_symbol,
                    side=exit_side,
                    entry=float(pos_now.entry),
                    pnl=float(fill["pnl"]),
                    capital=eq_after,
                    reason="BIAS_FLIP",
                )
                self._log_event(
                    symbol, "EXIT_BIAS_FLIP", bias, None, None,
                    float(pos_now.entry), float(pos_now.stop), float(pos_now.target),
                    float(pos_now.amount), fill.get("filled_price"), float(fill["pnl"]),
                    eq_after, "BIAS_FLIP"
                )
                return

        if bias not in ("BUY", "SELL"):
            self._log_event(
                symbol, "EVAL", bias, setup_valid, entry_signal,
                entry_price, stop, target, 0.0, None, None, equity, "BIAS_HOLD"
            )
            return

        if not setup_valid:
            self._log_event(
                symbol, "EVAL", bias, setup_valid, entry_signal,
                entry_price, stop, target, 0.0, None, None, equity, "SETUP_INVALID"
            )
            return

        if not entry_signal:
            self._log_event(
                symbol, "EVAL", bias, setup_valid, entry_signal,
                entry_price, stop, target, 0.0, None, None, equity,
                entry_reason or "ENTRY_SIGNAL_FALSE"
            )
            return

        if entry_price is None or stop is None or target is None:
            self._log_event(
                symbol, "EVAL", bias, setup_valid, entry_signal,
                entry_price, stop, target, 0.0, None, None, equity,
                "ENTRY_LEVELS_MISSING"
            )
            return

        if self.broker.get_position(symbol):
            self._log_event(
                symbol, "EVAL", bias, setup_valid, entry_signal,
                entry_price, stop, target, 0.0, None, None, equity, "POSITION_EXISTS"
            )
            return

        entry_price = float(entry_price)
        stop = float(stop)
        target = float(target)

        # Compute position size
        amount = 0.0
        if bias == "BUY":
            amount = self.risk.size_for_long(equity, entry_price, stop)
        elif bias == "SELL":
            amount = self.risk.size_for_short(equity, entry_price, stop)

        if amount <= 0:
            self._log_event(
                symbol, "EVAL", bias, setup_valid, entry_signal,
                entry_price, stop, target, 0.0, None, None, equity, "RISK_SIZING_ZERO"
            )
            return

        # Open position
        if bias == "BUY":
            fill = self.broker.open_long(symbol, amount, entry_price, stop, target)
        elif bias == "SELL":
            fill = self.broker.open_short(symbol, amount, entry_price, stop, target)

        if fill.get("ok"):
            filled_price = (
                fill.get("entry_price")
                or fill.get("filled_price")
                or entry_price
            )

            eq_after = self.broker.equity_usdt()
            self._log_line(
                event="TRADE",
                base_symbol=base_symbol,
                side=bias,
                entry=entry_price,
                pnl=0.0,
                capital=eq_after,
                reason="OPEN",
            )
            self._log_event(
                symbol, "TRADE_OPEN", bias, setup_valid, entry_signal,
                entry_price, stop, target, amount, filled_price, 0.0,
                eq_after, None
            )
        else:
            self._log_event(
                symbol, "EVAL", bias, setup_valid, entry_signal,
                entry_price, stop, target, amount, None, None, equity,
                str(fill.get("reason"))
            )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper trader ETH/BTC Multi-Timeframe")
    p.add_argument("--exchange", type=str, default="binance")
    p.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    p.add_argument("--poll", type=int, default=30)
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--log", type=str, default="logs/paper.log")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    symbols = tuple(s.strip().upper() for s in args.symbols.split(","))
    cfg = PaperConfig(
        exchange_id=args.exchange,
        symbols=symbols,
        poll_seconds=args.poll,
        log_path=args.log,
    )
    logger = _setup_logger(cfg.log_path)
    ex = CCXTExchange(cfg.exchange_id)
    broker = BrokerSim(starting_cash_usdt=args.cash)
    risk = RiskManager(
        RiskConfig(
            risk_pct=args.risk,
            fee_pct=broker.fee_pct,
            slippage_pct=broker.slippage_pct,
        )
    )
    trader = PaperTrader(cfg, ex, broker, risk, logger)

    print(f"=== Starting paper trader | Symbols: {cfg.symbols} | Cash: {broker.cash_usdt:.2f}\n")
    while True:
        for sym in cfg.symbols:
            try:
                trader.step(sym)
            except Exception as e:
                logger.exception(f"ERROR | {sym} | {e}")
                trader._log_event(
                    sym, "ERROR", None, None, None,
                    None, None, None, None, None, None,
                    broker.equity_usdt(), str(e)
                )
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
