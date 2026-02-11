import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src.strategies.multi_timeframe import generate_signal
from src.utils.data_loader import normalize_ohlcv

from paper.exchange_ccxt import CCXTExchange
from paper.broker_sim import BrokerSim
from paper.risk_manager import RiskConfig, RiskManager


@dataclass(frozen=True)
class PaperConfig:
    exchange_id: str = "binance"
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")

    # Fixed by framework
    tf_4h: str = "4h"
    tf_1h: str = "1h"
    tf_15m: str = "15m"

    limit_4h: int = 120
    limit_1h: int = 240
    limit_15m: int = 240

    poll_seconds: int = 30

    # Spot-like behavior: SELL signals close longs only
    close_on_sell_bias: bool = True

    # Output
    csv_path: str = "paper/logs/paper_trades.csv"
    log_path: str = "paper/logs/paper.log"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("paper_trader")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s,%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    if not has_stream:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if not has_file:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _ensure_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ts_utc",
                "symbol",
                "event",
                "bias",
                "setup_valid",
                "entry_signal",
                "entry",
                "stop",
                "target",
                "amount",
                "filled_price",
                "pnl_usd",
                "equity_usdt",
                "reason",
            ]
        )


def _append_csv(path: str, row: dict) -> None:
    _ensure_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                row.get("ts_utc"),
                row.get("symbol"),
                row.get("event"),
                row.get("bias"),
                row.get("setup_valid"),
                row.get("entry_signal"),
                row.get("entry"),
                row.get("stop"),
                row.get("target"),
                row.get("amount"),
                row.get("filled_price"),
                row.get("pnl_usd"),
                row.get("equity_usdt"),
                row.get("reason"),
            ]
        )


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
        # "BTC/USDT" -> "BTC"
        return str(symbol).split("/")[0].upper()

    @staticmethod
    def _fmt_px(px: float | None) -> str:
        return f"{float(px):.2f}" if isinstance(px, (int, float)) else "NA"

    def _log_line(
        self,
        event: str,
        base_symbol: str,
        side: str,
        entry: float | None,
        stop: float | None,
        target: float | None,
        pnl: float,
        capital: float,
        reason: str | None = None,
    ) -> None:
        msg = (
            f"{event} | {base_symbol} | {side} | "
            f"Entry: {self._fmt_px(entry)} | Stop: {self._fmt_px(stop)} | Target: {self._fmt_px(target)} | "
            f"PnL: {float(pnl):.2f} | Capital: {float(capital):.2f}"
        )
        if reason:
            msg = f"{msg} | Reason: {reason}"
        self.logger.info(msg)

    def _fetch_context(
        self, symbol: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_4h = self.ex.fetch_ohlcv_df(symbol, self.cfg.tf_4h, self.cfg.limit_4h)
        df_1h = self.ex.fetch_ohlcv_df(symbol, self.cfg.tf_1h, self.cfg.limit_1h)
        df_15m = self.ex.fetch_ohlcv_df(symbol, self.cfg.tf_15m, self.cfg.limit_15m)
        return normalize_ohlcv(df_4h), normalize_ohlcv(df_1h), normalize_ohlcv(df_15m)

    @staticmethod
    def _levels_from_signal(
        sig: dict, last_price: float
    ) -> tuple[float, float, float, str | None]:
        entry = None
        stop = None
        target = None
        reason = None

        entry_dict = sig.get("entry_15m") if isinstance(sig, dict) else None
        if isinstance(entry_dict, dict):
            entry = entry_dict.get("entry_price")
            stop = entry_dict.get("stop")
            target = entry_dict.get("target")
            reason = entry_dict.get("reason")

        # Logging fallbacks: always show prices
        if entry is None:
            entry = float(last_price)
        if stop is None:
            stop = float(entry)
        if target is None:
            target = float(entry)

        return float(entry), float(stop), float(target), reason

    def _csv_row(
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
        equity_usdt: float,
        reason: str | None,
    ) -> dict:
        return {
            "ts_utc": _utc_iso(),
            "symbol": symbol,
            "event": event,
            "bias": bias,
            "setup_valid": setup_valid,
            "entry_signal": entry_signal,
            "entry": entry,
            "stop": stop,
            "target": target,
            "amount": amount,
            "filled_price": filled_price,
            "pnl_usd": pnl_usd,
            "equity_usdt": equity_usdt,
            "reason": reason,
        }

    def step(self, symbol: str) -> None:
        last_price = self.ex.fetch_last_price(symbol)
        if last_price is None:
            return
        self.broker.set_last_price(symbol, last_price)

        base_symbol = self._base_symbol(symbol)

        did_trade_event = False

        # Snapshot position before any broker action (broker may delete it on exit)
        pos_before = self.broker.get_position(symbol)

        # Manage open position
        exit_fill = self.broker.check_stop_target(symbol)
        if exit_fill:
            eq = self.broker.equity_usdt()
            if pos_before:
                self._log_line(
                    event="TRADE",
                    base_symbol=base_symbol,
                    side="SELL",
                    entry=float(pos_before.entry),
                    stop=float(pos_before.stop),
                    target=float(pos_before.target),
                    pnl=float(exit_fill["pnl"]),
                    capital=float(eq),
                )
            else:
                self._log_line(
                    event="TRADE",
                    base_symbol=base_symbol,
                    side="SELL",
                    entry=float(last_price),
                    stop=None,
                    target=None,
                    pnl=float(exit_fill["pnl"]),
                    capital=float(eq),
                    reason=str(exit_fill.get("reason")),
                )
            did_trade_event = True
            _append_csv(
                self.cfg.csv_path,
                {
                    "ts_utc": _utc_iso(),
                    "symbol": symbol,
                    "event": f"EXIT_{exit_fill['reason']}",
                    "bias": None,
                    "setup_valid": None,
                    "entry_signal": None,
                    "entry": float(pos_before.entry) if pos_before else None,
                    "stop": float(pos_before.stop) if pos_before else None,
                    "target": float(pos_before.target) if pos_before else None,
                    "amount": float(pos_before.amount) if pos_before else None,
                    "filled_price": exit_fill["filled_price"],
                    "pnl_usd": exit_fill["pnl"],
                    "equity_usdt": eq,
                    "reason": exit_fill["reason"],
                },
            )

        # Compute signal
        df_4h, df_1h, df_15m = self._fetch_context(symbol)
        sig = generate_signal(df_4h, df_1h, df_15m)

        bias = sig["bias"]
        setup_valid = bool(sig["setup_valid"])
        entry_signal = bool(sig["entry_signal"])

        entry_px_eval, stop_eval, target_eval, reason = self._levels_from_signal(
            sig, last_price=float(last_price)
        )

        # Spot-like and SELL bias: close longs only
        if self.cfg.close_on_sell_bias and bias == "SELL":
            pos = self.broker.get_position(symbol)
            if pos:
                entry_snapshot = float(pos.entry)
                stop_snapshot = float(pos.stop)
                target_snapshot = float(pos.target)
                fill = self.broker.close_long(symbol, last_price, reason="SELL_BIAS")
                eq = self.broker.equity_usdt()
                self._log_line(
                    event="TRADE",
                    base_symbol=base_symbol,
                    side="SELL",
                    entry=entry_snapshot,
                    stop=stop_snapshot,
                    target=target_snapshot,
                    pnl=float(fill["pnl"]),
                    capital=float(eq),
                )
                did_trade_event = True
                _append_csv(
                    self.cfg.csv_path,
                    self._csv_row(
                        symbol=symbol,
                        event="CLOSE_ON_SELL_BIAS",
                        bias=bias,
                        setup_valid=setup_valid,
                        entry_signal=entry_signal,
                        entry=float(last_price),
                        stop=None,
                        target=None,
                        amount=float(pos.amount),
                        filled_price=float(fill["filled_price"]),
                        pnl_usd=float(fill["pnl"]),
                        equity_usdt=float(eq),
                        reason="SELL_BIAS_CLOSE_ONLY",
                    ),
                )

        # Entry check
        if bias != "BUY" or not setup_valid or not entry_signal:
            if not did_trade_event:
                self._log_line(
                    event="EVAL",
                    base_symbol=base_symbol,
                    side=str(bias),
                    entry=float(entry_px_eval),
                    stop=float(stop_eval),
                    target=float(target_eval),
                    pnl=0.0,
                    capital=float(self.broker.equity_usdt()),
                    reason=reason or "NO_VALID_ENTRY",
                )
            _append_csv(
                self.cfg.csv_path,
                self._csv_row(
                    symbol=symbol,
                    event="EVAL",
                    bias=bias,
                    setup_valid=setup_valid,
                    entry_signal=entry_signal,
                    entry=float(entry_px_eval),
                    stop=float(stop_eval),
                    target=float(target_eval),
                    amount=None,
                    filled_price=None,
                    pnl_usd=None,
                    equity_usdt=float(self.broker.equity_usdt()),
                    reason=reason or "NO_VALID_ENTRY",
                ),
            )
            return

        # BUY entry execution (spot long-only)
        if self.broker.get_position(symbol):
            if not did_trade_event:
                self._log_line(
                    event="EVAL",
                    base_symbol=base_symbol,
                    side=str(bias),
                    entry=float(entry_px_eval),
                    stop=float(stop_eval),
                    target=float(target_eval),
                    pnl=0.0,
                    capital=float(self.broker.equity_usdt()),
                    reason="POSITION_EXISTS",
                )
            _append_csv(
                self.cfg.csv_path,
                self._csv_row(
                    symbol=symbol,
                    event="EVAL",
                    bias=bias,
                    setup_valid=setup_valid,
                    entry_signal=entry_signal,
                    entry=float(entry_px_eval),
                    stop=float(stop_eval),
                    target=float(target_eval),
                    amount=None,
                    filled_price=None,
                    pnl_usd=None,
                    equity_usdt=float(self.broker.equity_usdt()),
                    reason="POSITION_EXISTS",
                ),
            )
            return

        entry_price = float(sig["entry_15m"].get("entry_price", last_price))
        stop = float(sig["entry_15m"]["stop"])
        target = float(sig["entry_15m"]["target"])

        equity = self.broker.equity_usdt()
        amount = self.risk.size_for_long(
            equity_usdt=equity, entry=entry_price, stop=stop
        )
        if amount <= 0:
            _append_csv(
                self.cfg.csv_path,
                self._csv_row(
                    symbol=symbol,
                    event="EVAL",
                    bias=bias,
                    setup_valid=setup_valid,
                    entry_signal=entry_signal,
                    entry=float(entry_price),
                    stop=float(stop),
                    target=float(target),
                    amount=0.0,
                    filled_price=None,
                    pnl_usd=None,
                    equity_usdt=float(equity),
                    reason="RISK_SIZING_ZERO",
                ),
            )
            return

        fill = self.broker.open_long(
            symbol, amount=amount, price=entry_price, stop=stop, target=target
        )
        if not fill.get("ok"):
            _append_csv(
                self.cfg.csv_path,
                self._csv_row(
                    symbol=symbol,
                    event="EVAL",
                    bias=bias,
                    setup_valid=setup_valid,
                    entry_signal=entry_signal,
                    entry=float(entry_price),
                    stop=float(stop),
                    target=float(target),
                    amount=float(amount),
                    filled_price=None,
                    pnl_usd=None,
                    equity_usdt=float(self.broker.equity_usdt()),
                    reason=str(fill.get("reason")),
                ),
            )
            return

        eq_after = self.broker.equity_usdt()
        self._log_line(
            event="TRADE",
            base_symbol=base_symbol,
            side="BUY",
            entry=float(entry_price),
            stop=float(stop),
            target=float(target),
            pnl=0.0,
            capital=float(eq_after),
        )
        did_trade_event = True
        _append_csv(
            self.cfg.csv_path,
            self._csv_row(
                symbol=symbol,
                event="TRADE_OPEN",
                bias=bias,
                setup_valid=setup_valid,
                entry_signal=entry_signal,
                entry=float(entry_price),
                stop=float(stop),
                target=float(target),
                amount=float(amount),
                filled_price=float(fill["filled_price"]),
                pnl_usd=0.0,
                equity_usdt=float(eq_after),
                reason=reason or "OPEN_LONG",
            ),
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paper trader (SPOT only, REST feed via ccxt)"
    )
    p.add_argument(
        "--exchange", type=str, default="binance", help="SPOT only (must be 'binance')"
    )
    p.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    p.add_argument("--poll", type=int, default=30)
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--out", type=str, default="paper/logs/paper_trades.csv")
    p.add_argument("--log", type=str, default="paper/logs/paper.log")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Paper trader is SPOT only. Guardrail against using futures/perps exchanges.
    exchange_id = str(args.exchange).strip().lower()
    if exchange_id != "binance":
        raise SystemExit("Paper trader is SPOT-only. Use: --exchange binance")

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    allowed = {"BTC/USDT", "ETH/USDT"}
    for s in symbols:
        if s not in allowed:
            raise SystemExit(f"Unsupported symbol {s}. Allowed: BTC/USDT, ETH/USDT")

    cfg = PaperConfig(
        exchange_id=exchange_id,
        symbols=symbols,
        poll_seconds=int(args.poll),
        csv_path=args.out,
        log_path=args.log,
    )

    logger = _setup_logger(cfg.log_path)

    ex = CCXTExchange(cfg.exchange_id)
    broker = BrokerSim(starting_cash_usdt=float(args.cash))
    risk = RiskManager(RiskConfig(risk_pct=float(args.risk)))

    print(
        f"Starting paper trader | Exchange: {cfg.exchange_id} | Symbols: {cfg.symbols} | Cash: {broker.cash_usdt:.2f}\n"
    )

    trader = PaperTrader(cfg, ex, broker, risk, logger)

    while True:
        for sym in cfg.symbols:
            try:
                trader.step(sym)
            except Exception as e:
                logger.exception(f"ERROR | {sym} | {e}")
                _append_csv(
                    cfg.csv_path,
                    {
                        "ts_utc": _utc_iso(),
                        "symbol": sym,
                        "event": "ERROR",
                        "bias": None,
                        "setup_valid": None,
                        "entry_signal": None,
                        "entry": None,
                        "stop": None,
                        "target": None,
                        "amount": None,
                        "filled_price": None,
                        "pnl_usd": None,
                        "equity_usdt": broker.equity_usdt(),
                        "reason": str(e),
                    },
                )
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
