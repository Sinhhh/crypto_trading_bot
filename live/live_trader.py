import argparse
import os
import time

from dataclasses import dataclass
from datetime import datetime, timezone

from src.strategies.multi_timeframe import generate_signal
from src.utils.data_loader import normalize_ohlcv
from paper.risk_manager import RiskConfig, RiskManager

from live.exchange import LiveExchange
from live.logging_utils import append_csv, setup_logger, utc_iso
from live.state import LivePosition, StateStore


@dataclass(frozen=True)
class LiveConfig:
    exchange_id: str = "binance"
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")

    tf_4h: str = "4h"
    tf_1h: str = "1h"
    tf_15m: str = "15m"

    limit_4h: int = 120
    limit_1h: int = 240
    limit_15m: int = 240

    poll_seconds: int = 30

    # Spot-only behavior
    close_on_sell_bias: bool = True

    # Files
    state_path: str = "live/state.json"
    log_path: str = "live/logs/live.log"
    trades_csv: str = "live/logs/live_trades.csv"

    # Safety
    live_trading_enabled: bool = False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live trader (SPOT only, Binance via ccxt)")
    p.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    p.add_argument("--poll", type=int, default=30)

    p.add_argument("--risk", type=float, default=0.01, help="Risk % of USDT balance")
    p.add_argument(
        "--dry-capital",
        type=float,
        default=1000.0,
        help="Used for sizing/logging in dry-run when balance is unavailable",
    )
    p.add_argument("--live", action="store_true", help="Enable real order placement")

    p.add_argument("--state", type=str, default="live/state.json")
    p.add_argument("--log", type=str, default="live/logs/live.log")
    p.add_argument("--out", type=str, default="live/logs/live_trades.csv")
    return p


def _base_symbol(symbol: str) -> str:
    return str(symbol).split("/")[0].upper()


def _log_line(
    logger,
    event: str,
    base: str,
    side: str,
    entry: float,
    stop: float,
    target: float,
    pnl: float,
    capital: float,
    reason: str | None = None,
) -> None:
    msg = (
        f"{event} | {base} | {side} | Entry: {entry:.2f} | Stop: {stop:.2f} | Target: {target:.2f} | "
        f"PnL: {pnl:.2f} | Capital: {capital:.2f}"
    )
    if reason:
        msg = f"{msg} | Reason: {reason}"
    logger.info(msg)


def _filled_price(order: dict, fallback: float) -> float:
    avg = order.get("average")
    if avg is not None:
        return float(avg)
    filled = order.get("filled")
    cost = order.get("cost")
    if filled and cost:
        try:
            return float(cost) / float(filled)
        except Exception:
            pass
    return float(fallback)


def main() -> None:
    args = _build_parser().parse_args()

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    allowed = {"BTC/USDT", "ETH/USDT"}
    for s in symbols:
        if s not in allowed:
            raise SystemExit(f"Unsupported symbol {s}. Allowed: BTC/USDT, ETH/USDT")

    cfg = LiveConfig(
        symbols=symbols,
        poll_seconds=int(args.poll),
        state_path=str(args.state),
        log_path=str(args.log),
        trades_csv=str(args.out),
        live_trading_enabled=bool(args.live),
    )

    logger = setup_logger(cfg.log_path)
    store = StateStore(cfg.state_path)
    positions = store.load()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if cfg.live_trading_enabled and (not api_key or not api_secret):
        raise SystemExit(
            "Missing BINANCE_API_KEY / BINANCE_API_SECRET (required for --live)"
        )

    ex = LiveExchange(cfg.exchange_id, api_key=api_key, api_secret=api_secret)
    risk = RiskManager(RiskConfig(risk_pct=float(args.risk)))

    mode = "LIVE" if cfg.live_trading_enabled else "DRY_RUN"
    logger.info(
        f"START | Mode: {mode} | Symbols: {cfg.symbols} | Poll: {cfg.poll_seconds}s"
    )
    # Create CSV output immediately so operators can confirm wiring.
    append_csv(
        cfg.trades_csv,
        {
            "ts_utc": utc_iso(),
            "symbol": "SYSTEM",
            "event": "START",
            "side": mode,
            "entry": None,
            "stop": None,
            "target": None,
            "amount": None,
            "filled_price": None,
            "pnl_usd": None,
            "capital_usdt": None,
            "reason": f"symbols={','.join(cfg.symbols)} poll={cfg.poll_seconds}",
        },
    )

    while True:
        for sym in cfg.symbols:
            base = _base_symbol(sym)
            try:
                last_price = ex.fetch_last_price(sym)
                if last_price is None:
                    continue

                base_asset, quote_asset = ex.base_quote(sym)
                if cfg.live_trading_enabled:
                    usdt_free = ex.fetch_free_balance(quote_asset)
                else:
                    # In dry-run, allow running without API keys (public data only).
                    usdt_free = float(args.dry_capital)

                # Fetch context
                df_4h = normalize_ohlcv(ex.fetch_ohlcv_df(sym, cfg.tf_4h, cfg.limit_4h))
                df_1h = normalize_ohlcv(ex.fetch_ohlcv_df(sym, cfg.tf_1h, cfg.limit_1h))
                df_15m = normalize_ohlcv(
                    ex.fetch_ohlcv_df(sym, cfg.tf_15m, cfg.limit_15m)
                )

                sig = generate_signal(df_4h, df_1h, df_15m)
                bias = sig["bias"]
                setup_valid = bool(sig["setup_valid"])
                entry_signal = bool(sig["entry_signal"])

                entry_dict = sig.get("entry_15m") if isinstance(sig, dict) else None
                if isinstance(entry_dict, dict):
                    entry_v = entry_dict.get("entry_price")
                    stop_v = entry_dict.get("stop")
                    target_v = entry_dict.get("target")
                    entry = float(entry_v) if entry_v is not None else float(last_price)
                    stop = float(stop_v) if stop_v is not None else float(entry)
                    target = float(target_v) if target_v is not None else float(entry)
                else:
                    entry = float(last_price)
                    stop = float(entry)
                    target = float(entry)
                reason = (
                    entry_dict.get("reason") if isinstance(entry_dict, dict) else None
                )

                # Manage open position (local state)
                pos = positions.get(sym)
                did_trade = False

                if pos:
                    # Stop/target exits
                    if float(last_price) <= float(pos.stop):
                        pnl = float(pos.amount) * (float(last_price) - float(pos.entry))
                        if cfg.live_trading_enabled:
                            order = ex.create_market_sell(sym, pos.amount)
                            fill_px = _filled_price(order, fallback=float(last_price))
                        else:
                            fill_px = float(last_price)

                        did_trade = True
                        _log_line(
                            logger,
                            "TRADE",
                            base,
                            "SELL",
                            pos.entry,
                            pos.stop,
                            pos.target,
                            pnl,
                            usdt_free,
                            reason="STOP",
                        )
                        append_csv(
                            cfg.trades_csv,
                            {
                                "ts_utc": utc_iso(),
                                "symbol": sym,
                                "event": "EXIT_STOP",
                                "side": "SELL",
                                "entry": float(pos.entry),
                                "stop": float(pos.stop),
                                "target": float(pos.target),
                                "amount": float(pos.amount),
                                "filled_price": float(fill_px),
                                "pnl_usd": float(pnl),
                                "capital_usdt": float(usdt_free),
                                "reason": "STOP",
                            },
                        )
                        positions.pop(sym, None)
                        store.save(positions)

                    elif float(last_price) >= float(pos.target):
                        pnl = float(pos.amount) * (float(last_price) - float(pos.entry))
                        if cfg.live_trading_enabled:
                            order = ex.create_market_sell(sym, pos.amount)
                            fill_px = _filled_price(order, fallback=float(last_price))
                        else:
                            fill_px = float(last_price)

                        did_trade = True
                        _log_line(
                            logger,
                            "TRADE",
                            base,
                            "SELL",
                            pos.entry,
                            pos.stop,
                            pos.target,
                            pnl,
                            usdt_free,
                            reason="TARGET",
                        )
                        append_csv(
                            cfg.trades_csv,
                            {
                                "ts_utc": utc_iso(),
                                "symbol": sym,
                                "event": "EXIT_TARGET",
                                "side": "SELL",
                                "entry": float(pos.entry),
                                "stop": float(pos.stop),
                                "target": float(pos.target),
                                "amount": float(pos.amount),
                                "filled_price": float(fill_px),
                                "pnl_usd": float(pnl),
                                "capital_usdt": float(usdt_free),
                                "reason": "TARGET",
                            },
                        )
                        positions.pop(sym, None)
                        store.save(positions)

                # SELL bias closes long (spot constraint)
                pos = positions.get(sym)
                if pos and cfg.close_on_sell_bias and bias == "SELL":
                    pnl = float(pos.amount) * (float(last_price) - float(pos.entry))
                    if cfg.live_trading_enabled:
                        order = ex.create_market_sell(sym, pos.amount)
                        fill_px = _filled_price(order, fallback=float(last_price))
                    else:
                        fill_px = float(last_price)

                    did_trade = True
                    _log_line(
                        logger,
                        "TRADE",
                        base,
                        "SELL",
                        pos.entry,
                        pos.stop,
                        pos.target,
                        pnl,
                        usdt_free,
                        reason="SELL_BIAS",
                    )
                    append_csv(
                        cfg.trades_csv,
                        {
                            "ts_utc": utc_iso(),
                            "symbol": sym,
                            "event": "CLOSE_ON_SELL_BIAS",
                            "side": "SELL",
                            "entry": float(pos.entry),
                            "stop": float(pos.stop),
                            "target": float(pos.target),
                            "amount": float(pos.amount),
                            "filled_price": float(fill_px),
                            "pnl_usd": float(pnl),
                            "capital_usdt": float(usdt_free),
                            "reason": "SELL_BIAS",
                        },
                    )
                    positions.pop(sym, None)
                    store.save(positions)

                # Entry (BUY only)
                if (
                    sym not in positions
                    and bias == "BUY"
                    and setup_valid
                    and entry_signal
                ):
                    amount = risk.size_for_long(
                        equity_usdt=float(usdt_free),
                        entry=float(entry),
                        stop=float(stop),
                    )
                    amount = ex.round_amount(sym, amount)
                    info = ex.market_info(sym)
                    if info.min_amount is not None and amount < float(info.min_amount):
                        amount = 0.0

                    if amount > 0 and cfg.live_trading_enabled:
                        order = ex.create_market_buy(sym, amount)
                        fill_px = _filled_price(order, fallback=float(entry))
                    elif amount > 0:
                        fill_px = float(entry)
                    else:
                        fill_px = None

                    if amount > 0 and fill_px is not None:
                        did_trade = True
                        positions[sym] = LivePosition(
                            symbol=sym,
                            amount=float(amount),
                            entry=float(fill_px),
                            stop=float(stop),
                            target=float(target),
                            opened_at=datetime.now(timezone.utc).isoformat(),
                        )
                        store.save(positions)

                        _log_line(
                            logger,
                            "TRADE",
                            base,
                            "BUY",
                            float(fill_px),
                            float(stop),
                            float(target),
                            0.0,
                            usdt_free,
                            reason="OPEN_LONG",
                        )
                        append_csv(
                            cfg.trades_csv,
                            {
                                "ts_utc": utc_iso(),
                                "symbol": sym,
                                "event": "TRADE_OPEN",
                                "side": "BUY",
                                "entry": float(fill_px),
                                "stop": float(stop),
                                "target": float(target),
                                "amount": float(amount),
                                "filled_price": float(fill_px),
                                "pnl_usd": 0.0,
                                "capital_usdt": float(usdt_free),
                                "reason": reason or "OPEN_LONG",
                            },
                        )

                # Always print EVAL when nothing traded
                if not did_trade:
                    _log_line(
                        logger,
                        "EVAL",
                        base,
                        str(bias),
                        float(entry),
                        float(stop),
                        float(target),
                        0.0,
                        float(usdt_free),
                        reason=reason or "NO_VALID_ENTRY",
                    )
                    append_csv(
                        cfg.trades_csv,
                        {
                            "ts_utc": utc_iso(),
                            "symbol": sym,
                            "event": "EVAL",
                            "side": str(bias),
                            "entry": float(entry),
                            "stop": float(stop),
                            "target": float(target),
                            "amount": None,
                            "filled_price": None,
                            "pnl_usd": 0.0,
                            "capital_usdt": float(usdt_free),
                            "reason": reason or "NO_VALID_ENTRY",
                        },
                    )

            except Exception as e:
                logger.exception(f"ERROR | {sym} | {e}")
                append_csv(
                    cfg.trades_csv,
                    {
                        "ts_utc": utc_iso(),
                        "symbol": sym,
                        "event": "ERROR",
                        "side": "",
                        "entry": None,
                        "stop": None,
                        "target": None,
                        "amount": None,
                        "filled_price": None,
                        "pnl_usd": None,
                        "capital_usdt": None,
                        "reason": str(e),
                    },
                )

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
