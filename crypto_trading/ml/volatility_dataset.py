from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone

import numpy as np
import pandas as pd

from crypto_trading.indicators.momentum import adx, rsi
from crypto_trading.regimes.regime_detector import detect_regime


@dataclass(frozen=True)
class VolatilityDatasetConfig:
    # Predict: will we see a "big move" within the next horizon?
    horizon_bars: int = 96

    # Label definition: max future abs return >= move_threshold => y=1
    # Example: 0.01 means a 1% move within horizon.
    move_threshold: float = 0.01

    # Feature windows
    sma_len: int = 50
    volume_sma_len: int = 20
    min_bars_required: int = 220

    # Intraday features
    bar_minutes: int = 15

    # Speed knob
    sample_step: int = 1


def _to_utc_timestamp(ts) -> pd.Timestamp | None:
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
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


def _ensure_features(df: pd.DataFrame, cfg: VolatilityDatasetConfig) -> pd.DataFrame:
    df = df.copy()

    if "_sma_gate" not in df.columns:
        df["_sma_gate"] = df["close"].rolling(int(cfg.sma_len)).mean()

    if "_vol_sma20" not in df.columns:
        df["_vol_sma20"] = df["volume"].rolling(int(cfg.volume_sma_len)).mean()

    if "_rsi14" not in df.columns:
        df["_rsi14"] = rsi(df["close"].astype(float), 14)

    if "_adx14" not in df.columns:
        df["_adx14"] = adx(df, 14)

    return df


def _feature_row(
    df_sig: pd.DataFrame,
    *,
    symbol: str,
    regime: str,
    source: str,
    ts,
    bar_minutes: int,
) -> dict:
    last = df_sig.iloc[-1]
    close_v = float(last["close"])
    vol_v = float(last["volume"])

    ret_1 = float(df_sig["close"].pct_change(1).iloc[-1])
    ret_3 = float(df_sig["close"].pct_change(3).iloc[-1]) if len(df_sig) >= 4 else np.nan
    ret_6 = float(df_sig["close"].pct_change(6).iloc[-1]) if len(df_sig) >= 7 else np.nan
    rv_20 = float(df_sig["close"].pct_change().rolling(20).std().iloc[-1])

    sma_gate_v = float(last.get("_sma_gate", np.nan))
    vol_sma_v = float(last.get("_vol_sma20", np.nan))
    rsi_v = float(last.get("_rsi14", np.nan))
    adx_v = float(last.get("_adx14", np.nan))

    dist_sma = (close_v / sma_gate_v - 1.0) if (sma_gate_v and not np.isnan(sma_gate_v)) else np.nan
    vol_ratio = (vol_v / vol_sma_v) if (vol_sma_v and not np.isnan(vol_sma_v)) else np.nan

    ts_utc = _to_utc_timestamp(ts)
    bar_of_day = _bar_of_day(ts_utc, bar_minutes=int(bar_minutes))
    day_of_week = _day_of_week(ts_utc)
    bars_to_midnight = _bars_to_utc_midnight(ts_utc, bar_minutes=int(bar_minutes))

    return {
        "symbol": str(symbol),
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


def build_big_move_dataset(
    df: pd.DataFrame,
    *,
    cfg: VolatilityDatasetConfig,
    symbol: str,
) -> pd.DataFrame:
    """Build a dataset for "will we see a big move soon".

    Label:
      y=1 if max_{k=1..horizon} |close[i+k]/close[i] - 1| >= move_threshold

    Features:
      Similar schema to tp/sl gate so inference can share infrastructure.

    Returns a DataFrame with columns: features + y + time
    """
    if df is None or len(df) < int(cfg.min_bars_required) + int(cfg.horizon_bars) + 5:
        return pd.DataFrame()

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = _ensure_features(df, cfg)

    closes = df["close"].astype(float).to_numpy()
    times = df.index

    horizon = int(max(1, cfg.horizon_bars))
    step = int(max(1, cfg.sample_step))

    rows: list[dict] = []

    for i in range(int(cfg.min_bars_required), len(df) - horizon, step):
        df_sig = df.iloc[:i]
        if len(df_sig) < int(cfg.min_bars_required):
            continue

        entry_close = float(closes[i - 1])
        if not np.isfinite(entry_close) or entry_close <= 0:
            continue

        future = closes[i : i + horizon]
        if len(future) != horizon:
            continue

        fut_ret = future / entry_close - 1.0
        max_abs_ret = float(np.nanmax(np.abs(fut_ret)))
        if not np.isfinite(max_abs_ret):
            continue

        y = 1 if max_abs_ret >= float(cfg.move_threshold) else 0

        # We keep regime/source as coarse context features.
        try:
            regime = str(detect_regime(df_sig)).upper()
        except Exception:
            regime = "TRANSITION"
        source = "volatility"

        feat = _feature_row(
            df_sig,
            symbol=str(symbol),
            regime=regime,
            source=source,
            ts=times[i - 1],
            bar_minutes=int(cfg.bar_minutes),
        )
        feat["y"] = int(y)
        feat["time"] = times[i - 1]
        rows.append(feat)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("time").reset_index(drop=True)
    return out
