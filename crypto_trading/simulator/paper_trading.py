from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ccxt
import pandas as pd

from crypto_trading.config.config import get_api_key, get_secret_key
from crypto_trading.indicators.momentum import adx, rsi
from crypto_trading.indicators.volatility import atr
from crypto_trading.indicators.moving_averages import ema
from crypto_trading.lifecycle.spot_lifecycle import SpotTradeLifecycle, SpotTradeState
from crypto_trading.unified.unified_signal import (
    unified_signal_mtf,
    unified_signal_with_meta,
)
from crypto_trading.utils.logging import enable_sigpipe_default, print_trade_row
from crypto_trading.utils.paths import ensure_dir, reports_dir


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}/{quote}"
    if s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        return f"{base}/USDT"
    return s


def _ohlcv_to_df(ohlcv: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df.set_index("datetime")
    return df


def _ensure_indicators(df: pd.DataFrame, *, sma_len: int) -> None:
    # In live mode we keep the window bounded (limit=N), so recomputing once per poll is OK.
    # We still avoid recomputing multiple times within the same poll.
    if "_sma_gate" not in df.columns:
        df["_sma_gate"] = df["close"].rolling(int(sma_len)).mean()
    if "_vol_sma20" not in df.columns:
        df["_vol_sma20"] = df["volume"].rolling(20).mean()
    if "_rsi14" not in df.columns:
        df["_rsi14"] = rsi(df["close"].astype(float), 14)
    if "_adx14" not in df.columns:
        df["_adx14"] = adx(df, 14)
    if "atr_14" not in df.columns:
        df["atr_14"] = atr(df, 14)


@dataclass
class PaperConfig:
    starting_balance: float = 100.0
    position_usdt: float = 25.0
    max_risk_per_trade: float = 0.03
    fee_rate: float = 0.0
    slippage_rate: float = 0.0001
    theoretical_max: bool = False

    # gates
    personality_mode: str = "GROWTH"
    safe_threshold: float = 0.80
    growth_threshold: float = 0.55
    volume_multiplier: float = 1.2
    sma_len: int = 50

    min_bars_required: int = 220

    # optional win-rate focused filters (opt-in)
    allowed_regimes: tuple[str, ...] | None = None
    min_adx_for_buy: float | None = None
    min_atr_ratio_for_buy: float | None = None
    htf_trend_for_buy: bool = False
    htf_ema_len: int = 200
    min_htf_bars: int = 50


class PaperTrader:
    def __init__(
        self, *, symbol: str, entry_tf: Literal["15m", "1h", "4h"], cfg: PaperConfig
    ):
        self.symbol = str(symbol)
        self.entry_tf = entry_tf
        self.cfg = cfg

        self.cash_usdt = float(cfg.starting_balance)
        self.asset_qty = 0.0

        self.state = SpotTradeState()
        self.lifecycle = SpotTradeLifecycle(
            atr_multiplier=2.0,
            trail_pct=None,
            trail_atr_mult=2.5,
            partial_profit_pct=0.05,
            partial_sell_fraction=0.3,
            max_bars_in_trade=48,
            cooldown_bars=3,
        )

        self.last_regime: str = "TRANSITION"
        self.last_source: str = "none"

        self._active_candle_open_ts = None
        self._bar_index = 0

        self._eps_qty = 1e-12
        self._eps_usdt = 1e-9

    def _fee(self, amount_usdt: float) -> float:
        return float(amount_usdt) * float(self.cfg.fee_rate)

    def _buy_fill(self, px: float) -> float:
        return float(px) * (1.0 + float(self.cfg.slippage_rate))

    def _sell_fill(self, px: float) -> float:
        return float(px) * (1.0 - float(self.cfg.slippage_rate))

    def _threshold(self) -> float:
        mode = str(self.cfg.personality_mode).upper()
        return float(
            self.cfg.safe_threshold if mode == "SAFE" else self.cfg.growth_threshold
        )

    def _assert_invariants(self, *, context: str) -> None:
        if self.cash_usdt < -self._eps_usdt:
            raise RuntimeError(
                f"[SAFETY] Negative cash ({self.cash_usdt}) at {context}"
            )
        if self.asset_qty < -self._eps_qty:
            raise RuntimeError(
                f"[SAFETY] Negative asset ({self.asset_qty}) at {context}"
            )

        if self.state.in_position and self.asset_qty <= self._eps_qty:
            raise RuntimeError(
                f"[SAFETY] Desync: in_position=True but asset_qty={self.asset_qty} at {context}"
            )
        if (not self.state.in_position) and self.asset_qty > self._eps_qty:
            raise RuntimeError(
                f"[SAFETY] Desync: in_position=False but asset_qty={self.asset_qty} at {context}"
            )

    def _estimate_confidence(self, df: pd.DataFrame, *, source: str) -> float:
        if df is None or len(df) < 30:
            return 0.0
        try:
            if source == "mean_reversion":
                r_val = (
                    float(df["_rsi14"].iloc[-1])
                    if "_rsi14" in df.columns
                    else float(rsi(df["close"], 14).iloc[-1])
                )
                if pd.isna(r_val):
                    return 0.0
                return max(0.0, min(1.0, abs(r_val - 50.0) / 25.0))

            a_val = (
                float(df["_adx14"].iloc[-1])
                if "_adx14" in df.columns
                else float(adx(df, 14).iloc[-1])
            )
            if pd.isna(a_val):
                return 0.0
            return max(0.0, min(1.0, a_val / 50.0))
        except Exception:
            return 0.0

    def _volume_ok(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < 20:
            return False
        avg_vol = (
            df["_vol_sma20"].iloc[-1]
            if "_vol_sma20" in df.columns
            else df["volume"].rolling(20).mean().iloc[-1]
        )
        latest = df["volume"].iloc[-1]
        if pd.isna(avg_vol) or pd.isna(latest):
            return False
        return float(latest) > float(avg_vol) * float(self.cfg.volume_multiplier)

    def step(
        self,
        *,
        df_entry: pd.DataFrame,
        df_1h: pd.DataFrame | None,
        df_4h: pd.DataFrame | None,
        report: "TradeReport",
        print_rows: bool,
    ) -> None:
        """One poll step.

        - Detects new candle open time
        - Executes signal decisions on new candle (prev candle close -> exec at open)
        - Manages intrabar stops using current candle's evolving high/low
        """
        if df_entry is None or len(df_entry) < max(3, int(self.cfg.min_bars_required)):
            return

        _ensure_indicators(df_entry, sma_len=int(self.cfg.sma_len))

        candle_open_ts = df_entry.index[-1]
        is_new_candle = candle_open_ts != self._active_candle_open_ts

        if is_new_candle:
            self._active_candle_open_ts = candle_open_ts
            self._bar_index += 1

        # Execution price: open at start of new candle, otherwise mark at last close
        if is_new_candle:
            exec_price = float(df_entry["open"].iloc[-1])
        else:
            exec_price = float(df_entry["close"].iloc[-1])

        # Decision slice uses only CLOSED candles (everything up to previous row)
        signal = "HOLD"
        regime = self.last_regime
        source = self.last_source

        if is_new_candle:
            df_sig = df_entry.iloc[:-1]

            # Trend filter on entry timeframe
            sma_val = float(df_sig["_sma_gate"].iloc[-1])

            if self.entry_tf == "15m":
                if df_1h is None or df_4h is None:
                    signal, regime, source = "HOLD", "TRANSITION", "none"
                else:
                    now_ts = df_sig.index[-1]
                    df_1h_sig = df_1h.loc[:now_ts]
                    df_4h_sig = df_4h.loc[:now_ts]
                    signal, regime, source = unified_signal_mtf(
                        df_sig, df_1h_sig, df_4h_sig
                    )
            elif self.entry_tf in {"1h", "4h"}:
                signal, regime, source = unified_signal_with_meta(df_sig)
            else:
                raise ValueError(f"Unsupported entry_tf={self.entry_tf}")

            # Optional regime filter
            if signal == "BUY" and self.cfg.allowed_regimes is not None:
                allowed = {str(x).strip().upper() for x in self.cfg.allowed_regimes}
                if str(regime).strip().upper() not in allowed:
                    signal = "HOLD"

            # Optional HTF trend filter (15m bot only: uses 4h context already fetched)
            if signal == "BUY" and bool(self.cfg.htf_trend_for_buy):
                if self.entry_tf != "15m" or df_4h is None:
                    signal = "HOLD"
                else:
                    try:
                        now_ts = df_sig.index[-1]
                        df_4h_sig = df_4h.loc[:now_ts]
                        if df_4h_sig is None or len(df_4h_sig) < int(self.cfg.min_htf_bars):
                            signal = "HOLD"
                        else:
                            ema_len = int(self.cfg.htf_ema_len)
                            ema_4h = ema(df_4h_sig["close"].astype(float), ema_len).iloc[-1]
                            close_4h = float(df_4h_sig["close"].iloc[-1])
                            if pd.isna(ema_4h) or pd.isna(close_4h) or float(close_4h) < float(ema_4h):
                                signal = "HOLD"
                    except Exception:
                        signal = "HOLD"

            # Gate BUY only
            if signal == "BUY":
                if pd.isna(sma_val) or float(df_sig["close"].iloc[-1]) < float(sma_val):
                    signal = "HOLD"
                elif self.cfg.min_atr_ratio_for_buy is not None:
                    try:
                        atr_val = float(df_sig["atr_14"].iloc[-1])
                        close_val = float(df_sig["close"].iloc[-1])
                        if pd.isna(atr_val) or pd.isna(close_val) or close_val <= 0:
                            signal = "HOLD"
                        else:
                            atr_ratio = float(atr_val) / float(close_val)
                            if float(atr_ratio) < float(self.cfg.min_atr_ratio_for_buy):
                                signal = "HOLD"
                    except Exception:
                        signal = "HOLD"
                elif self.cfg.min_adx_for_buy is not None:
                    try:
                        a_val = float(df_sig["_adx14"].iloc[-1])
                        if pd.isna(a_val) or float(a_val) < float(self.cfg.min_adx_for_buy):
                            signal = "HOLD"
                    except Exception:
                        signal = "HOLD"
                elif not self._volume_ok(df_sig):
                    signal = "HOLD"
                else:
                    conf = self._estimate_confidence(df_sig, source=source)
                    if conf < self._threshold():
                        signal = "HOLD"

            self.last_regime = str(regime)
            self.last_source = str(source)

        # Optional dynamic ATR stop derived from CLOSED candles only.
        atr_stop_override = None
        if is_new_candle and signal == "BUY" and (not self.state.in_position):
            try:
                df_sig = df_entry.iloc[:-1]
                if len(df_sig) >= 30:
                    _ensure_indicators(df_sig, sma_len=int(self.cfg.sma_len))
                    atr_val = float(df_sig["atr_14"].iloc[-1])
                    recent_vol = df_sig["close"].pct_change().rolling(20).std().iloc[-1]
                    if pd.isna(recent_vol):
                        atr_mult = float(self.lifecycle.atr_multiplier)
                    elif float(recent_vol) < 0.005:
                        atr_mult = 1.5
                    else:
                        atr_mult = 3.0
                    if (not pd.isna(atr_val)) and atr_val > 0:
                        atr_stop_override = float(exec_price) - float(atr_mult) * float(
                            atr_val
                        )
            except Exception:
                atr_stop_override = None

        # Feed lifecycle a management window including the current (possibly in-progress) candle
        df_mgmt = df_entry

        self._assert_invariants(context="start step")

        self.state, trade_event = self.lifecycle.update(
            df_1h=df_mgmt,
            state=self.state,
            signal=str(signal),
            regime=str(regime),
            price=float(exec_price),
            bar_index=int(self._bar_index),
            atr_stop_override=atr_stop_override,
        )

        if trade_event is None:
            return

        etype = str(trade_event.get("type", ""))
        event_time = df_entry.index[-1]

        if etype == "ENTRY":
            entry_price = self._buy_fill(
                float(trade_event.get("entry_price", exec_price))
            )

            stop_price = trade_event.get("atr_stop", None)
            if stop_price is None:
                stop_price = getattr(self.state, "atr_stop", None)
            stop_price = float(stop_price) if stop_price is not None else None

            stop_dist = (entry_price - stop_price) if stop_price is not None else 0.0

            target_notional = min(float(self.cfg.position_usdt), float(self.cash_usdt))
            conf = (
                self._estimate_confidence(df_entry.iloc[:-1], source=self.last_source)
                if len(df_entry) > 1
                else 0.0
            )
            qty_multiplier = 0.5 + conf
            target_notional = min(
                float(self.cash_usdt), float(target_notional) * float(qty_multiplier)
            )

            target_qty = (target_notional / entry_price) if entry_price > 0 else 0.0

            risk_usdt = float(self.cash_usdt) * float(self.cfg.max_risk_per_trade)
            risk_qty = (
                (risk_usdt / stop_dist) if (stop_dist and stop_dist > 0) else target_qty
            )
            qty = max(0.0, min(float(target_qty), float(risk_qty)))

            if qty <= self._eps_qty or entry_price <= 0:
                self.state = SpotTradeState(cooldown_until=self.state.cooldown_until)
                return

            cost = qty * entry_price
            total_cost = cost + self._fee(cost)

            if total_cost > self.cash_usdt + self._eps_usdt:
                denom = entry_price * (1.0 + float(self.cfg.fee_rate))
                affordable_qty = (self.cash_usdt / denom) if denom > 0 else 0.0
                qty = min(qty, max(0.0, affordable_qty))
                cost = qty * entry_price
                total_cost = cost + self._fee(cost)

            if qty <= self._eps_qty or total_cost > self.cash_usdt + self._eps_usdt:
                self.state = SpotTradeState(cooldown_until=self.state.cooldown_until)
                return

            self.cash_usdt -= total_cost
            self.asset_qty += qty

            self.state.qty = qty
            self.state.entry_notional_usdt = cost

            self._assert_invariants(context="after ENTRY")

            report.write(
                {
                    "time": event_time.isoformat(),
                    "symbol": self.symbol,
                    "event": "ENTRY",
                    "price": round(entry_price, 8),
                    "qty": round(qty, 12),
                    "fee_usdt": round(self._fee(cost), 8),
                    "pnl_usdt": 0.0,
                    "cash_usdt": round(self.cash_usdt, 8),
                    "asset_qty": round(self.asset_qty, 12),
                    "regime": str(self.last_regime),
                    "source": str(self.last_source),
                }
            )
            if print_rows:
                print_trade_row(
                    event_time,
                    self.symbol,
                    "BUY",
                    0.0,
                    cash=self.cash_usdt,
                    asset=self.asset_qty,
                )

        elif etype == "PARTIAL":
            exec_px = self._sell_fill(float(trade_event.get("price", exec_price)))
            qty_sold = float(trade_event.get("qty_sold") or 0.0)
            qty_sold = max(0.0, min(qty_sold, self.asset_qty))

            if qty_sold <= self._eps_qty or exec_px <= 0:
                return

            proceeds = qty_sold * exec_px
            fee_paid = self._fee(proceeds)
            self.cash_usdt += proceeds - fee_paid
            self.asset_qty -= qty_sold

            self._assert_invariants(context="after PARTIAL")

            pnl_partial = float(trade_event.get("pnl_usdt") or 0.0)
            report.write(
                {
                    "time": event_time.isoformat(),
                    "symbol": self.symbol,
                    "event": "PARTIAL",
                    "price": round(exec_px, 8),
                    "qty": round(qty_sold, 12),
                    "fee_usdt": round(fee_paid, 8),
                    "pnl_usdt": round(pnl_partial, 8),
                    "cash_usdt": round(self.cash_usdt, 8),
                    "asset_qty": round(self.asset_qty, 12),
                    "regime": str(self.last_regime),
                    "source": str(self.last_source),
                }
            )
            if print_rows:
                print_trade_row(
                    event_time,
                    self.symbol,
                    "SELL_PARTIAL",
                    pnl_partial,
                    cash=self.cash_usdt,
                    asset=self.asset_qty,
                )

        elif etype == "EXIT":
            exec_px = self._sell_fill(float(trade_event.get("price", exec_price)))
            qty_to_sell = float(trade_event.get("qty") or 0.0)
            qty_to_sell = max(0.0, min(qty_to_sell, self.asset_qty))

            if qty_to_sell > self._eps_qty and exec_px > 0:
                proceeds = qty_to_sell * exec_px
                fee_paid = self._fee(proceeds)
                self.cash_usdt += proceeds - fee_paid
                self.asset_qty -= qty_to_sell
            else:
                fee_paid = 0.0

            self._assert_invariants(context="after EXIT")

            pnl_leg = float(trade_event.get("pnl_usdt") or 0.0)
            pnl_total = float(trade_event.get("pnl_total_usdt") or pnl_leg)

            report.write(
                {
                    "time": event_time.isoformat(),
                    "symbol": self.symbol,
                    "event": "EXIT",
                    "price": round(exec_px, 8),
                    "qty": round(qty_to_sell, 12),
                    "fee_usdt": round(fee_paid, 8),
                    "pnl_usdt": round(pnl_total, 8),
                    "cash_usdt": round(self.cash_usdt, 8),
                    "asset_qty": round(self.asset_qty, 12),
                    "regime": str(self.last_regime),
                    "source": str(self.last_source),
                }
            )
            if print_rows:
                print_trade_row(
                    event_time,
                    self.symbol,
                    "SELL_EXIT",
                    pnl_leg,
                    cash=self.cash_usdt,
                    asset=self.asset_qty,
                )


class TradeReport:
    def __init__(self, *, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._init_file()

    def _init_file(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "time",
                    "symbol",
                    "event",
                    "price",
                    "qty",
                    "fee_usdt",
                    "pnl_usdt",
                    "cash_usdt",
                    "asset_qty",
                    "regime",
                    "source",
                ],
            )
            w.writeheader()

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "time",
                    "symbol",
                    "event",
                    "price",
                    "qty",
                    "fee_usdt",
                    "pnl_usdt",
                    "cash_usdt",
                    "asset_qty",
                    "regime",
                    "source",
                ],
            )
            w.writerow(row)


def _make_exchange() -> ccxt.Exchange:
    # Keys optional for public endpoints; keep for future real trading.
    return ccxt.mexc(
        {
            "timeout": 30000,
            "enableRateLimit": True,
            "apiKey": get_api_key(),
            "secret": get_secret_key(),
        }
    )


def _fetch_window(
    exchange: ccxt.Exchange, *, symbol: str, timeframe: str, limit: int
) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=int(limit))
    return _ohlcv_to_df(ohlcv)


def main() -> None:
    enable_sigpipe_default()

    p = argparse.ArgumentParser(
        description="Paper trading (simulated) using live OHLCV from MEXC"
    )
    p.add_argument(
        "--symbol", default="ETHUSDT", help="Symbol like ETHUSDT or ETH/USDT"
    )
    p.add_argument("--timeframe", default="1h", help="Timeframe: 15m, 1h, 4h")
    p.add_argument(
        "--entry-tf",
        default=None,
        choices=["15m", "1h", "4h"],
        help="Bot selection (default inferred from timeframe)",
    )
    p.add_argument("--poll-seconds", type=float, default=20.0, help="Polling interval")
    p.add_argument(
        "--limit", type=int, default=500, help="Candles to keep in the rolling window"
    )

    p.add_argument("--starting-balance", type=float, default=100.0)
    p.add_argument("--position-usdt", type=float, default=25.0)
    p.add_argument("--max-risk", type=float, default=0.03)

    p.add_argument("--fee-rate", type=float, default=0.0)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--theoretical-max", action="store_true")

    p.add_argument("--print-rows", action="store_true", help="Print BUY/SELL rows")

    # Optional trade filters
    p.add_argument(
        "--allowed-regimes",
        default=None,
        help="Comma-separated regimes allowed for BUY (e.g. TREND_UP,CROSS_UP)",
    )
    p.add_argument(
        "--min-adx-for-buy",
        type=float,
        default=None,
        help="Avoid ranges: require ADX(14) >= this value",
    )
    p.add_argument(
        "--min-atr-ratio-for-buy",
        type=float,
        default=None,
        help="High-vol only: require atr_14/close >= this value (e.g. 0.01)",
    )
    p.add_argument(
        "--htf-trend-for-buy",
        action="store_true",
        help="15m bot only: require 4h close above 4h EMA",
    )
    p.add_argument(
        "--htf-ema-len",
        type=int,
        default=200,
        help="EMA length for HTF trend filter (default 200)",
    )

    args = p.parse_args()

    tf = str(args.timeframe).strip().lower().replace("min", "m")

    entry_tf: Literal["15m", "1h", "4h"]
    if args.entry_tf is not None:
        entry_tf = args.entry_tf
    else:
        if "15" in tf:
            entry_tf = "15m"
        elif "4h" in tf or "240" in tf:
            entry_tf = "4h"
        else:
            entry_tf = "1h"

    cfg = PaperConfig(
        starting_balance=float(args.starting_balance),
        position_usdt=float(args.position_usdt),
        max_risk_per_trade=float(args.max_risk),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        theoretical_max=bool(args.theoretical_max),
        allowed_regimes=(
            tuple(x.strip().upper() for x in str(args.allowed_regimes).split(",") if x.strip())
            if args.allowed_regimes
            else None
        ),
        min_adx_for_buy=(float(args.min_adx_for_buy) if args.min_adx_for_buy is not None else None),
        min_atr_ratio_for_buy=(
            float(args.min_atr_ratio_for_buy)
            if args.min_atr_ratio_for_buy is not None
            else None
        ),
        htf_trend_for_buy=bool(args.htf_trend_for_buy),
        htf_ema_len=int(args.htf_ema_len),
    )
    if cfg.theoretical_max:
        cfg.fee_rate = 0.0
        cfg.slippage_rate = 0.0

    symbol_ccxt = _normalize_symbol(args.symbol)
    symbol_report = symbol_ccxt.replace("/", "")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir(f"paper_trades_live_{symbol_report}_{tf}_{ts}.csv")
    report = TradeReport(path=report_path)

    exchange = _make_exchange()
    trader = PaperTrader(symbol=symbol_report, entry_tf=entry_tf, cfg=cfg)

    print(f"Paper trading (SIM) {symbol_ccxt} tf={tf} entry_tf={entry_tf}")
    print(f"Report: {report_path}")
    print(
        f"Mode: {'THEORETICAL_MAX' if cfg.theoretical_max else 'REALISTIC'} | fee_rate={cfg.fee_rate} | slippage_rate={cfg.slippage_rate}"
    )

    while True:
        try:
            df_entry = _fetch_window(
                exchange, symbol=symbol_ccxt, timeframe=tf, limit=int(args.limit)
            )

            df_1h = None
            df_4h = None
            if entry_tf == "15m":
                df_1h = _fetch_window(
                    exchange,
                    symbol=symbol_ccxt,
                    timeframe="1h",
                    limit=max(300, int(args.limit)),
                )
                df_4h = _fetch_window(
                    exchange,
                    symbol=symbol_ccxt,
                    timeframe="4h",
                    limit=max(300, int(args.limit)),
                )

            trader.step(
                df_entry=df_entry,
                df_1h=df_1h,
                df_4h=df_4h,
                report=report,
                print_rows=bool(args.print_rows),
            )

            time.sleep(float(args.poll_seconds))
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as e:
            # Keep the loop alive for transient network issues
            print(f"Error: {e}")
            time.sleep(max(5.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
