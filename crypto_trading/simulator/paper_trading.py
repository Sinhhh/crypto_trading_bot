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

from crypto_trading.config.config import (
    get_api_key,
    get_secret_key,
    load_execution_config_from_env,
    load_gate_config_from_env,
    parse_allowed_entry_regimes,
    parse_allowed_regimes,
)
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
    allowed_entry_regimes: dict[str, bool] | None = None
    allow_mean_reversion_in_range: bool = True
    min_adx_for_buy: float | None = None
    min_atr_ratio_for_buy: float | None = None
    htf_trend_for_buy: bool = False
    htf_ema_len: int = 200
    min_htf_bars: int = 50

    # Optional ML gate model (joblib path). If set, replaces heuristic confidence.
    ml_model_path: str | None = None

    # Optional volatility (should-trade) model (joblib path).
    vol_model_path: str | None = None
    vol_threshold: float | None = None

    # Optional day-trading rules (UTC-day boundary)
    daytrade_utc_flat: bool = False
    no_entry_last_bars: int | None = None

    # Optional: only allow CROSS_UP/DOWN trades if confirmed by HTF or strong volume.
    cross_require_htf_or_volume: bool = False
    cross_volume_multiplier: float = 1.5
    cross_volume_sma_len: int = 20


class PaperTrader:
    def __init__(
        self, *, symbol: str, entry_tf: Literal["5m", "15m", "1h", "4h"], cfg: PaperConfig
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

        self._ml_gate = None
        if cfg.ml_model_path:
            try:
                from crypto_trading.ml.gate_model import load_gate_model

                self._ml_gate = load_gate_model(str(cfg.ml_model_path))
            except Exception:
                self._ml_gate = None

        self._vol_gate = None
        if cfg.vol_model_path:
            try:
                from crypto_trading.ml.volatility_model import load_volatility_model

                self._vol_gate = load_volatility_model(str(cfg.vol_model_path))
            except Exception:
                self._vol_gate = None

    def _bar_minutes(self) -> int:
        tf = str(self.entry_tf).lower().replace("min", "m")
        if tf == "5m":
            return 5
        if tf == "15m":
            return 15
        if tf == "4h":
            return 240
        return 60

    def _bars_to_utc_midnight(self, ts) -> float:
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize(timezone.utc)
            else:
                t = t.tz_convert(timezone.utc)
            next_midnight = t.normalize() + pd.Timedelta(days=1)
            delta_min = (next_midnight - t).total_seconds() / 60.0
            return float(delta_min / float(max(1, self._bar_minutes())))
        except Exception:
            return float("nan")

    def _is_last_bar_of_utc_day(self, ts) -> bool:
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize(timezone.utc)
            else:
                t = t.tz_convert(timezone.utc)
            next_midnight = t.normalize() + pd.Timedelta(days=1)
            return (t + pd.Timedelta(minutes=int(self._bar_minutes()))) >= next_midnight
        except Exception:
            return False

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

    def _ml_confidence(self, df: pd.DataFrame, *, regime: str, source: str, exec_ts=None) -> float:
        if self._ml_gate is None:
            return self._estimate_confidence(df, source=source)

        if df is None or len(df) < 30:
            return 0.0

        try:
            last = df.iloc[-1]
            close_v = float(last["close"])
            vol_v = float(last["volume"])

            ret_1 = float(df["close"].pct_change(1).iloc[-1])
            ret_3 = float(df["close"].pct_change(3).iloc[-1]) if len(df) >= 4 else float("nan")
            ret_6 = float(df["close"].pct_change(6).iloc[-1]) if len(df) >= 7 else float("nan")
            rv_20 = float(df["close"].pct_change().rolling(20).std().iloc[-1])

            sma_gate_v = float(last.get("_sma_gate", float("nan")))
            vol_sma_v = float(last.get("_vol_sma20", float("nan")))
            rsi_v = float(last.get("_rsi14", float("nan")))
            adx_v = float(last.get("_adx14", float("nan")))

            dist_sma = (close_v / sma_gate_v - 1.0) if (sma_gate_v and not pd.isna(sma_gate_v)) else float("nan")
            vol_ratio = (vol_v / vol_sma_v) if (vol_sma_v and not pd.isna(vol_sma_v)) else float("nan")

            # Intraday time features (UTC)
            ts = exec_ts
            if ts is None and len(df) > 0:
                ts = df.index[-1]
            try:
                t = pd.Timestamp(ts) if ts is not None else None
                if t is None:
                    ts_utc = None
                elif t.tzinfo is None:
                    ts_utc = t.tz_localize(timezone.utc)
                else:
                    ts_utc = t.tz_convert(timezone.utc)
            except Exception:
                ts_utc = None

            minute_of_day = (
                (int(ts_utc.hour) * 60 + int(ts_utc.minute)) if ts_utc is not None else None
            )
            bar_of_day = (
                float(int(minute_of_day) // max(1, self._bar_minutes()))
                if minute_of_day is not None
                else float("nan")
            )
            day_of_week = float(int(ts_utc.dayofweek)) if ts_utc is not None else float("nan")
            bars_to_midnight = self._bars_to_utc_midnight(ts) if ts is not None else float("nan")

            features = {
                "symbol": str(self.symbol),
                "bar_of_day": bar_of_day,
                "day_of_week": day_of_week,
                "bars_to_utc_midnight": bars_to_midnight,
                "close": close_v,
                "volume": vol_v,
                "ret_1": ret_1,
                "ret_3": ret_3,
                "ret_6": ret_6,
                "rv_20": rv_20,
                "rsi14": rsi_v,
                "adx14": adx_v,
                "dist_sma": dist_sma,
                "vol_ratio": vol_ratio,
                "regime": str(regime),
                "source": str(source),
            }
            return float(self._ml_gate.predict_proba_row(features))
        except Exception:
            return self._estimate_confidence(df, source=source)

    def _vol_confidence(self, df: pd.DataFrame, *, regime: str, source: str, exec_ts=None) -> float | None:
        if self._vol_gate is None:
            return None
        if df is None or len(df) < 30:
            return 0.0

        try:
            last = df.iloc[-1]
            close_v = float(last["close"])
            vol_v = float(last["volume"])

            ret_1 = float(df["close"].pct_change(1).iloc[-1])
            ret_3 = float(df["close"].pct_change(3).iloc[-1]) if len(df) >= 4 else float("nan")
            ret_6 = float(df["close"].pct_change(6).iloc[-1]) if len(df) >= 7 else float("nan")
            rv_20 = float(df["close"].pct_change().rolling(20).std().iloc[-1])

            sma_gate_v = float(last.get("_sma_gate", float("nan")))
            vol_sma_v = float(last.get("_vol_sma20", float("nan")))
            rsi_v = float(last.get("_rsi14", float("nan")))
            adx_v = float(last.get("_adx14", float("nan")))

            dist_sma = (close_v / sma_gate_v - 1.0) if (sma_gate_v and not pd.isna(sma_gate_v)) else float("nan")
            vol_ratio = (vol_v / vol_sma_v) if (vol_sma_v and not pd.isna(vol_sma_v)) else float("nan")

            ts = exec_ts
            if ts is None and len(df) > 0:
                ts = df.index[-1]

            try:
                t = pd.Timestamp(ts) if ts is not None else None
                if t is None:
                    ts_utc = None
                elif t.tzinfo is None:
                    ts_utc = t.tz_localize(timezone.utc)
                else:
                    ts_utc = t.tz_convert(timezone.utc)
            except Exception:
                ts_utc = None

            minute_of_day = (int(ts_utc.hour) * 60 + int(ts_utc.minute)) if ts_utc is not None else None
            bar_of_day = float(int(minute_of_day) // max(1, self._bar_minutes())) if minute_of_day is not None else float("nan")
            day_of_week = float(int(ts_utc.dayofweek)) if ts_utc is not None else float("nan")
            bars_to_midnight = self._bars_to_utc_midnight(ts) if ts is not None else float("nan")

            features = {
                "symbol": str(self.symbol),
                "bar_of_day": bar_of_day,
                "day_of_week": day_of_week,
                "bars_to_utc_midnight": bars_to_midnight,
                "close": close_v,
                "volume": vol_v,
                "ret_1": ret_1,
                "ret_3": ret_3,
                "ret_6": ret_6,
                "rv_20": rv_20,
                "rsi14": rsi_v,
                "adx14": adx_v,
                "dist_sma": dist_sma,
                "vol_ratio": vol_ratio,
                "regime": str(regime),
                "source": str(source),
            }
            return float(self._vol_gate.predict_proba_row(features))
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
            elif self.entry_tf in {"5m", "1h", "4h"}:
                # If we have 4h context (fetched for 15m bot), we can use it as HTF confirmation.
                df_htf_sig = None
                if df_4h is not None:
                    try:
                        now_ts = df_sig.index[-1]
                        df_htf_sig = df_4h.loc[:now_ts]
                    except Exception:
                        df_htf_sig = None

                signal, regime, source = unified_signal_with_meta(
                    df_sig,
                    df_htf=df_htf_sig,
                    cross_require_htf_or_volume=bool(self.cfg.cross_require_htf_or_volume),
                    cross_volume_multiplier=float(self.cfg.cross_volume_multiplier),
                    cross_volume_sma_len=int(self.cfg.cross_volume_sma_len),
                )
            else:
                raise ValueError(f"Unsupported entry_tf={self.entry_tf}")

            # Day-trading force-flat (UTC day): exit on the last bar of the UTC day.
            if bool(self.cfg.daytrade_utc_flat) and self.state.in_position:
                exec_ts = df_entry.index[-1]
                if self._is_last_bar_of_utc_day(exec_ts):
                    signal = "SELL"

            # Optional regime filter
            if signal == "BUY":
                regime_u = str(regime).strip().upper()
                source_u = str(source).strip().lower()

                # Volatility gate (should-trade): require a predicted big move soon.
                if self.cfg.vol_threshold is not None:
                    vprob = self._vol_confidence(df_sig, regime=regime, source=source, exec_ts=df_entry.index[-1])
                    if vprob is None or float(vprob) < float(self.cfg.vol_threshold):
                        signal = "HOLD"

                # Optional explicit permission table (preferred when set)
                if self.cfg.allowed_entry_regimes is not None:
                    if (
                        regime_u == "RANGE"
                        and bool(self.cfg.allow_mean_reversion_in_range)
                        and source_u == "mean_reversion"
                    ):
                        pass
                    else:
                        if not bool(self.cfg.allowed_entry_regimes.get(regime_u, False)):
                            signal = "HOLD"

                # Legacy allowlist
                elif self.cfg.allowed_regimes is not None:
                    allowed = {str(x).strip().upper() for x in self.cfg.allowed_regimes}
                    if regime_u not in allowed:
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
                    # Day-trading entry cut-off: no new entries in last N bars of UTC day.
                    if self.cfg.no_entry_last_bars is not None:
                        exec_ts = df_entry.index[-1]
                        btm = self._bars_to_utc_midnight(exec_ts)
                        if (not pd.isna(btm)) and float(btm) <= float(self.cfg.no_entry_last_bars):
                            signal = "HOLD"

                    if signal == "BUY":
                        conf = self._ml_confidence(
                            df_sig,
                            regime=regime,
                            source=source,
                            exec_ts=df_entry.index[-1],
                        )
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
                self._ml_confidence(
                    df_entry.iloc[:-1],
                    regime=self.last_regime,
                    source=self.last_source,
                    exec_ts=event_time,
                )
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

    p.add_argument(
        "--fee-rate",
        type=float,
        default=None,
        help="Override env FEE_RATE (default uses env/config)",
    )
    p.add_argument(
        "--slippage-rate",
        type=float,
        default=None,
        help="Override env SLIPPAGE_RATE (default uses env/config)",
    )
    p.add_argument(
        "--theoretical-max",
        action="store_true",
        default=None,
        help="Override env THEORETICAL_MAX (if enabled: fee/slippage forced to 0)",
    )

    p.add_argument("--print-rows", action="store_true", help="Print BUY/SELL rows")

    p.add_argument(
        "--ml-model-path",
        default=None,
        help="Optional joblib model path for ML gating (TP-before-SL probability)",
    )

    p.add_argument(
        "--vol-model-path",
        default=None,
        help="Optional joblib model path for volatility gating (should-trade probability)",
    )
    p.add_argument(
        "--vol-threshold",
        type=float,
        default=None,
        help="If --vol-model-path (or VOL_MODEL_PATH) is set: require vol_prob >= this threshold",
    )

    # Optional trade filters
    p.add_argument(
        "--allowed-regimes",
        default=None,
        help="Comma-separated regimes allowed for BUY (e.g. TREND_UP,CROSS_UP)",
    )
    p.add_argument(
        "--allowed-entry-regimes",
        default=None,
        help="Explicit regime permission table for BUY like: TREND_UP=1,RANGE=0,CROSS_UP=0,TRANSITION=0",
    )
    p.add_argument(
        "--allow-mean-reversion-in-range",
        action="store_true",
        default=None,
        help="If using --allowed-entry-regimes, still allow BUY when source=mean_reversion in RANGE",
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
        default=None,
        help="15m bot only: require 4h close above 4h EMA",
    )
    p.add_argument(
        "--htf-ema-len",
        type=int,
        default=None,
        help="EMA length for HTF trend filter (default uses env/config)",
    )

    p.add_argument(
        "--cross-require-htf-or-volume",
        action="store_true",
        default=None,
        help="Only allow CROSS_UP/DOWN trades if confirmed by HTF regime or extra-strong volume",
    )
    p.add_argument(
        "--cross-volume-multiplier",
        type=float,
        default=None,
        help="If CROSS confirmation uses volume: require vol > SMA(vol)*multiplier (default uses env/config)",
    )
    p.add_argument(
        "--cross-volume-sma-len",
        type=int,
        default=None,
        help="SMA length for CROSS volume confirmation (default uses env/config)",
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

    exec_cfg = load_execution_config_from_env()
    gate_cfg = load_gate_config_from_env()

    fee_rate = float(args.fee_rate) if args.fee_rate is not None else float(exec_cfg.fee_rate)
    slippage_rate = (
        float(args.slippage_rate)
        if args.slippage_rate is not None
        else float(exec_cfg.slippage_rate)
    )
    theoretical_max = (
        bool(args.theoretical_max)
        if args.theoretical_max is not None
        else bool(exec_cfg.theoretical_max)
    )

    allowed_regimes = (
        parse_allowed_regimes(args.allowed_regimes)
        if args.allowed_regimes
        else parse_allowed_regimes(gate_cfg.allowed_regimes)
    )
    allowed_entry_regimes = (
        parse_allowed_entry_regimes(args.allowed_entry_regimes)
        if args.allowed_entry_regimes
        else parse_allowed_entry_regimes(gate_cfg.allowed_entry_regimes)
    )
    allow_mean_reversion_in_range = (
        bool(args.allow_mean_reversion_in_range)
        if args.allow_mean_reversion_in_range is not None
        else bool(gate_cfg.allow_mean_reversion_in_range)
    )

    min_adx_for_buy = (
        float(args.min_adx_for_buy)
        if args.min_adx_for_buy is not None
        else gate_cfg.min_adx_for_buy
    )
    min_atr_ratio_for_buy = (
        float(args.min_atr_ratio_for_buy)
        if args.min_atr_ratio_for_buy is not None
        else gate_cfg.min_atr_ratio_for_buy
    )

    htf_trend_for_buy = (
        bool(args.htf_trend_for_buy)
        if args.htf_trend_for_buy is not None
        else bool(gate_cfg.htf_trend_for_buy)
    )
    htf_ema_len = int(args.htf_ema_len) if args.htf_ema_len is not None else int(gate_cfg.htf_ema_len)

    cross_require_htf_or_volume = (
        bool(args.cross_require_htf_or_volume)
        if args.cross_require_htf_or_volume is not None
        else bool(gate_cfg.cross_require_htf_or_volume)
    )
    cross_volume_multiplier = (
        float(args.cross_volume_multiplier)
        if args.cross_volume_multiplier is not None
        else float(gate_cfg.cross_volume_multiplier)
    )
    cross_volume_sma_len = (
        int(args.cross_volume_sma_len)
        if args.cross_volume_sma_len is not None
        else int(gate_cfg.cross_volume_sma_len)
    )

    ml_model_path = str(args.ml_model_path) if args.ml_model_path else gate_cfg.ml_model_path
    vol_model_path = str(args.vol_model_path) if args.vol_model_path else gate_cfg.vol_model_path
    vol_threshold = float(args.vol_threshold) if args.vol_threshold is not None else gate_cfg.vol_threshold

    cfg = PaperConfig(
        starting_balance=float(args.starting_balance),
        position_usdt=float(args.position_usdt),
        max_risk_per_trade=float(args.max_risk),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        theoretical_max=theoretical_max,
        personality_mode=str(gate_cfg.personality_mode),
        safe_threshold=float(gate_cfg.safe_threshold),
        growth_threshold=float(gate_cfg.growth_threshold),
        volume_multiplier=float(gate_cfg.volume_multiplier),
        sma_len=int(gate_cfg.sma_len),
        allowed_regimes=allowed_regimes,
        allowed_entry_regimes=allowed_entry_regimes,
        allow_mean_reversion_in_range=allow_mean_reversion_in_range,
        min_adx_for_buy=min_adx_for_buy,
        min_atr_ratio_for_buy=min_atr_ratio_for_buy,
        htf_trend_for_buy=htf_trend_for_buy,
        htf_ema_len=htf_ema_len,
        min_htf_bars=int(gate_cfg.min_htf_bars),
        ml_model_path=ml_model_path,
        vol_model_path=vol_model_path,
        vol_threshold=vol_threshold,
        cross_require_htf_or_volume=cross_require_htf_or_volume,
        cross_volume_multiplier=cross_volume_multiplier,
        cross_volume_sma_len=cross_volume_sma_len,
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
            elif bool(cfg.cross_require_htf_or_volume):
                # For 1H/4H bots: fetch 4H context so CROSS confirmation can use HTF regime.
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
