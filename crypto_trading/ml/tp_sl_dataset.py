from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone

import numpy as np
import pandas as pd

from crypto_trading.indicators.momentum import adx, rsi
from crypto_trading.indicators.volatility import atr
from crypto_trading.lifecycle.spot_lifecycle import SpotTradeLifecycle, SpotTradeState
from crypto_trading.unified.unified_signal import unified_signal_with_meta


@dataclass(frozen=True)
class TpSlDatasetConfig:
    horizon_bars: int = 24
    atr_len: int = 14
    sma_len: int = 50
    volume_sma_len: int = 20
    volume_multiplier: float = 1.2
    min_bars_required: int = 220

    # Match backtest lifecycle defaults (tune to your live config)
    lifecycle_atr_multiplier: float = 2.2
    lifecycle_trail_pct: float | None = None
    lifecycle_trail_atr_mult: float | None = 3.8
    lifecycle_partial_profit_pct: float = 0.14
    lifecycle_partial_sell_fraction: float = 0.2
    lifecycle_max_bars_in_trade: int = 96
    lifecycle_cooldown_bars: int = 6
    lifecycle_exit_on_regime_change: bool = True
    lifecycle_exit_on_sell_signal: bool = True

    # Entry stop override logic (matches backtest: scale ATR mult by recent vol)
    dynamic_atr_stop: bool = True
    low_vol_threshold: float = 0.005
    low_vol_atr_mult: float = 1.5
    high_vol_atr_mult: float = 3.0

    # What to do when neither TP nor SL is hit within horizon.
    # - "drop": exclude the sample
    # - "zero": keep sample with y=0
    unresolved: str = "drop"

    # Speed knob: only evaluate every Nth candidate bar (1 = use all).
    sample_step: int = 1

    # --- Intraday / day-trading features ---
    # Bar duration in minutes (e.g. 60 for 1h, 15 for 15m, 5 for 5m)
    bar_minutes: int = 60


def _to_utc_timestamp(ts) -> pd.Timestamp | None:
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            # Assume naive timestamps are UTC
            return t.tz_localize(timezone.utc)
        return t.tz_convert(timezone.utc)
    except Exception:
        return None


def _bar_of_day(ts_utc: pd.Timestamp | None, *, bar_minutes: int) -> float:
    if ts_utc is None:
        return float("nan")
    try:
        minutes = int(ts_utc.hour) * 60 + int(ts_utc.minute)
        return float(minutes // max(1, int(bar_minutes)))
    except Exception:
        return float("nan")


def _day_of_week(ts_utc: pd.Timestamp | None) -> float:
    if ts_utc is None:
        return float("nan")
    try:
        return float(int(ts_utc.dayofweek))
    except Exception:
        return float("nan")


def _bars_to_utc_midnight(ts_utc: pd.Timestamp | None, *, bar_minutes: int) -> float:
    if ts_utc is None:
        return float("nan")
    try:
        bar_minutes = max(1, int(bar_minutes))
        next_midnight = ts_utc.normalize() + pd.Timedelta(days=1)
        delta_min = (next_midnight - ts_utc).total_seconds() / 60.0
        return float(delta_min / float(bar_minutes))
    except Exception:
        return float("nan")


def _ensure_features(df: pd.DataFrame, cfg: TpSlDatasetConfig) -> pd.DataFrame:
    df = df.copy()

    if "_sma_gate" not in df.columns:
        df["_sma_gate"] = df["close"].rolling(int(cfg.sma_len)).mean()

    if "_vol_sma" not in df.columns:
        df["_vol_sma"] = df["volume"].rolling(int(cfg.volume_sma_len)).mean()

    if "_rsi14" not in df.columns:
        df["_rsi14"] = rsi(df["close"].astype(float), 14)

    if "_adx14" not in df.columns:
        df["_adx14"] = adx(df, 14)

    atr_col = f"atr_{int(cfg.atr_len)}"
    if atr_col not in df.columns:
        df[atr_col] = atr(df, int(cfg.atr_len))

    return df


def _compute_feature_row(
    df_sig: pd.DataFrame,
    *,
    symbol: str | None,
    regime: str,
    source: str,
    entry_ts,
    bar_minutes: int,
) -> dict:
    """Compute features using ONLY df_sig (closed candles up to decision time)."""
    last = df_sig.iloc[-1]

    close = float(last["close"])
    vol = float(last["volume"])

    # Price returns
    ret_1 = float(df_sig["close"].pct_change(1).iloc[-1])
    ret_3 = float(df_sig["close"].pct_change(3).iloc[-1]) if len(df_sig) >= 4 else np.nan
    ret_6 = float(df_sig["close"].pct_change(6).iloc[-1]) if len(df_sig) >= 7 else np.nan

    # Realized volatility
    rv_20 = float(df_sig["close"].pct_change().rolling(20).std().iloc[-1])

    # Gates / indicators already precomputed
    sma_gate = float(last.get("_sma_gate", np.nan))
    vol_sma = float(last.get("_vol_sma", np.nan))
    rsi14 = float(last.get("_rsi14", np.nan))
    adx14 = float(last.get("_adx14", np.nan))

    # Ratios
    dist_sma = (close / sma_gate - 1.0) if (sma_gate and not np.isnan(sma_gate)) else np.nan
    vol_ratio = (vol / vol_sma) if (vol_sma and not np.isnan(vol_sma)) else np.nan

    ts_utc = _to_utc_timestamp(entry_ts)
    bar_of_day = _bar_of_day(ts_utc, bar_minutes=int(bar_minutes))
    day_of_week = _day_of_week(ts_utc)
    bars_to_midnight = _bars_to_utc_midnight(ts_utc, bar_minutes=int(bar_minutes))

    return {
        "symbol": (str(symbol) if symbol is not None else ""),
        "bar_of_day": bar_of_day,
        "day_of_week": day_of_week,
        "bars_to_utc_midnight": bars_to_midnight,
        "close": close,
        "volume": vol,
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_6": ret_6,
        "rv_20": rv_20,
        "rsi14": rsi14,
        "adx14": adx14,
        "dist_sma": dist_sma,
        "vol_ratio": vol_ratio,
        "regime": str(regime),
        "source": str(source),
    }


def _entry_atr_stop_override(
    df_sig: pd.DataFrame, *, cfg: TpSlDatasetConfig, entry_price: float
) -> float | None:
    """Replicate backtest entry ATR stop override using only closed candles."""
    try:
        if df_sig is None or len(df_sig) < 30:
            return None

        atr_col = f"atr_{int(cfg.atr_len)}"
        atr_val = float(df_sig[atr_col].iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return None

        if not bool(cfg.dynamic_atr_stop):
            atr_mult = float(cfg.lifecycle_atr_multiplier)
        else:
            recent_vol = float(df_sig["close"].pct_change().rolling(20).std().iloc[-1])
            if pd.isna(recent_vol):
                atr_mult = float(cfg.lifecycle_atr_multiplier)
            elif float(recent_vol) < float(cfg.low_vol_threshold):
                atr_mult = float(cfg.low_vol_atr_mult)
            else:
                atr_mult = float(cfg.high_vol_atr_mult)

        return float(entry_price) - float(atr_mult) * float(atr_val)
    except Exception:
        return None


def _simulate_label_via_lifecycle(
    df: pd.DataFrame,
    *,
    entry_bar_index: int,
    entry_regime: str,
    cfg: TpSlDatasetConfig,
) -> int | None:
    """Label via forward simulation.

        y=1 if lifecycle emits PARTIAL within horizon.
        Otherwise, when EXIT happens within horizon:
            - y=1 if the exit is profitable (pnl_total_usdt > 0)
            - y=0 otherwise
    None if unresolved.

    Important: uses the same decision timing as backtest:
    - decision on closed candle (df[:bar])
    - execute at bar open (df.open[bar])
    """
    lifecycle = SpotTradeLifecycle(
        atr_multiplier=float(cfg.lifecycle_atr_multiplier),
        trail_pct=cfg.lifecycle_trail_pct,
        trail_atr_mult=cfg.lifecycle_trail_atr_mult,
        partial_profit_pct=float(cfg.lifecycle_partial_profit_pct),
        partial_sell_fraction=float(cfg.lifecycle_partial_sell_fraction),
        max_bars_in_trade=int(cfg.lifecycle_max_bars_in_trade),
        cooldown_bars=int(cfg.lifecycle_cooldown_bars),
        exit_on_regime_change=bool(cfg.lifecycle_exit_on_regime_change),
        exit_on_sell_signal=bool(cfg.lifecycle_exit_on_sell_signal),
    )

    state = SpotTradeState()

    # Entry happens at bar `entry_bar_index` open.
    i = int(entry_bar_index)
    if i <= 0 or i >= len(df):
        return None

    df_sig = df.iloc[:i]  # closed candles up to i-1
    df_mgmt = df.iloc[: i + 1]  # include current bar for intrabar high/low
    entry_price = float(df["open"].iloc[i])

    atr_stop_override = _entry_atr_stop_override(df_sig, cfg=cfg, entry_price=entry_price)

    state, ev = lifecycle.update(
        df_1h=df_mgmt,
        state=state,
        signal="BUY",
        regime=str(entry_regime),
        price=float(entry_price),
        bar_index=i,
        atr_stop_override=atr_stop_override,
    )

    if ev is None or str(ev.get("type")) != "ENTRY":
        return None

    # Mimic backtest: qty is set after the entry is executed.
    state.qty = 1.0

    end = min(len(df) - 1, i + int(cfg.horizon_bars))
    for j in range(i + 1, end + 1):
        df_sig_j = df.iloc[:j]  # closed candles up to j-1
        sig, reg, src = unified_signal_with_meta(df_sig_j)
        # Execute that decision at bar j open
        df_mgmt_j = df.iloc[: j + 1]
        px = float(df["open"].iloc[j])

        state, ev = lifecycle.update(
            df_1h=df_mgmt_j,
            state=state,
            signal=str(sig),
            regime=str(reg),
            price=float(px),
            bar_index=int(j),
            atr_stop_override=None,
        )

        if ev is None:
            continue

        et = str(ev.get("type", ""))
        if et == "PARTIAL":
            return 1

        if et == "EXIT":
            pnl_total = ev.get("pnl_total_usdt", None)
            if pnl_total is None:
                # Fallback: compare exit price to entry (qty cancels out).
                exit_px = float(ev.get("price") or px)
                entry_px = float(state.entry_price or entry_price)
                pnl_total = exit_px - entry_px
            return 1 if float(pnl_total) > 0.0 else 0

    return None


def build_tp_before_sl_dataset(
    df_in: pd.DataFrame,
    *,
    cfg: TpSlDatasetConfig = TpSlDatasetConfig(),
    symbol: str | None = None,
) -> pd.DataFrame:
    """Build a supervised dataset for 1H long entries.

    Sampling logic:
    - Entry candidates come from your real strategy signal (BUY)
    - Applies the same always-on gates (SMA + volume)
    - Label is computed by simulating your SpotTradeLifecycle forward for `horizon_bars`
      and returning whether PARTIAL occurs before EXIT.

    Returns a DataFrame with feature columns + `y` label and metadata.
    """
    if df_in is None or len(df_in) < int(cfg.min_bars_required) + int(cfg.horizon_bars) + 2:
        return pd.DataFrame()

    df = _ensure_features(df_in, cfg)

    atr_col = f"atr_{int(cfg.atr_len)}"
    rows: list[dict] = []

    # idx is the execution bar index (entry happens at df.open[idx])
    step = max(1, int(getattr(cfg, "sample_step", 1)))
    for idx in range(int(cfg.min_bars_required), len(df) - int(cfg.horizon_bars), step):
        df_sig = df.iloc[:idx]
        if len(df_sig) < int(cfg.min_bars_required):
            continue

        signal, regime, source = unified_signal_with_meta(df_sig)
        if signal != "BUY":
            continue

        # Apply the same always-on gates (except ML confidence)
        last = df_sig.iloc[-1]
        sma_gate = float(last.get("_sma_gate", np.nan))
        if np.isnan(sma_gate) or float(last["close"]) < sma_gate:
            continue

        vol_sma = float(last.get("_vol_sma", np.nan))
        if np.isnan(vol_sma) or vol_sma <= 0:
            continue
        if float(last["volume"]) <= vol_sma * float(cfg.volume_multiplier):
            continue

        # Ensure ATR is present for entry stop override.
        atr_val = float(last.get(atr_col, np.nan))
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        y = _simulate_label_via_lifecycle(
            df,
            entry_bar_index=int(idx),
            entry_regime=str(regime),
            cfg=cfg,
        )

        if y is None:
            if cfg.unresolved == "zero":
                y = 0
            else:
                continue

        feat = _compute_feature_row(
            df_sig,
            symbol=symbol,
            regime=regime,
            source=source,
            entry_ts=df.index[idx],
            bar_minutes=int(cfg.bar_minutes),
        )
        feat.update(
            {
                "y": int(y),
                "time": df.index[idx],
                "entry_index": int(idx),
                "regime": str(regime),
                "source": str(source),
            }
        )
        rows.append(feat)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Drop obvious NaNs from rolling stats
    out = out.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return out
