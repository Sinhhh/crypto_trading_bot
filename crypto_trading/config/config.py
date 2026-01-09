"""Configuration helpers.

Do NOT hardcode exchange keys in source control.

Set env vars instead:
- MEXC_API_KEY
- MEXC_SECRET_KEY
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


def get_api_key() -> str:
    return str(os.getenv("MEXC_API_KEY", ""))


def get_secret_key() -> str:
    return str(os.getenv("MEXC_SECRET_KEY", ""))


def _env_str(name: str) -> str | None:
    v = os.getenv(name, "").strip()
    return v if v else None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v == "":
        return bool(default)
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_float(name: str) -> float | None:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _env_int(name: str) -> int | None:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return int(float(v))
    except Exception:
        return None


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution model settings (affect Net PnL)."""

    fee_rate: float = 0.0
    slippage_rate: float = 0.0001
    theoretical_max: bool = False


@dataclass(frozen=True)
class GateConfig:
    """Entry gating / quality filters."""

    # ML gate
    ml_model_path: str | None = None
    # Volatility (should-trade) model gate
    vol_model_path: str | None = None
    vol_threshold: float | None = None
    personality_mode: str = "GROWTH"
    safe_threshold: float = 0.80
    growth_threshold: float = 0.55

    # Heuristic filters
    volume_multiplier: float = 1.2
    sma_len: int = 50
    min_adx_for_buy: float | None = None
    min_atr_ratio_for_buy: float | None = None

    # HTF alignment (used in backtest; paper trading currently supports 15m bot only)
    htf_trend_for_buy: bool = False
    htf_tf: str | None = None
    htf_ema_len: int = 200
    min_htf_bars: int = 50

    # CROSS_UP/DOWN quality filter
    cross_require_htf_or_volume: bool = False
    cross_volume_multiplier: float = 1.5
    cross_volume_sma_len: int = 20

    # Explicit regime permission table (default-deny style)
    allowed_regimes: str | tuple[str, ...] | list[str] | set[str] | None = None
    allowed_entry_regimes: str | dict[str, bool] | None = None
    allow_mean_reversion_in_range: bool = True


def parse_allowed_regimes(value: Any) -> tuple[str, ...] | None:
    """Parse allowed regime list input.

    Accepts either:
    - A CSV string like "TREND_UP,CROSS_UP"
    - An iterable of regime names
    Returns a tuple of uppercased strings or None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [x.strip().upper() for x in value.split(",") if x.strip()]
        return tuple(parts) if parts else None
    try:
        parts = [str(x).strip().upper() for x in value if str(x).strip()]
        return tuple(parts) if parts else None
    except Exception:
        return None


def parse_allowed_entry_regimes(value: Any) -> dict[str, bool] | None:
    """Parse explicit regime permission table.

    Accepts either:
    - A dict mapping regime->bool
    - A CSV string like: "TREND_UP=1,RANGE=0,CROSS_UP=0,TRANSITION=0"
    Returns a normalized dict with uppercased keys or None.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        out = {str(k).strip().upper(): bool(v) for k, v in value.items()}
        return out if out else None

    s = str(value).strip()
    if not s:
        return None

    tbl: dict[str, bool] = {}
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        kk = str(k).strip().upper()
        vv = str(v).strip().lower()
        if not kk:
            continue
        tbl[kk] = vv in {"1", "true", "yes", "y", "on"}

    return tbl if tbl else None


def load_execution_config_from_env(*, defaults: ExecutionConfig | None = None) -> ExecutionConfig:
    """Load execution settings from env vars.

    Env vars:
    - FEE_RATE
    - SLIPPAGE_RATE
    - THEORETICAL_MAX
    """
    d = defaults or ExecutionConfig()
    fee = _env_float("FEE_RATE")
    slip = _env_float("SLIPPAGE_RATE")
    theo = _env_bool("THEORETICAL_MAX", default=d.theoretical_max)
    return ExecutionConfig(
        fee_rate=float(fee) if fee is not None else float(d.fee_rate),
        slippage_rate=float(slip) if slip is not None else float(d.slippage_rate),
        theoretical_max=bool(theo),
    )


def load_gate_config_from_env(*, defaults: GateConfig | None = None) -> GateConfig:
    """Load gate settings from env vars.

    Common env vars:
    - ML_MODEL_PATH
    - VOL_MODEL_PATH, VOL_THRESHOLD
    - PERSONALITY_MODE, SAFE_ML_THRESHOLD, GROWTH_ML_THRESHOLD
    - VOLUME_MULTIPLIER, SMA_LEN, MIN_ADX_FOR_BUY, MIN_ATR_RATIO_FOR_BUY
    - HTF_TREND_FOR_BUY, HTF_TF, HTF_EMA_LEN, MIN_HTF_BARS
    - CROSS_REQUIRE_HTF_OR_VOLUME, CROSS_VOLUME_MULTIPLIER, CROSS_VOLUME_SMA_LEN
    - ALLOWED_REGIMES, ALLOWED_ENTRY_REGIMES, ALLOW_MEAN_REVERSION_IN_RANGE
    """
    d = defaults or GateConfig()

    ml_path = _env_str("ML_MODEL_PATH")
    vol_path = _env_str("VOL_MODEL_PATH")
    vol_th = _env_float("VOL_THRESHOLD")
    personality = _env_str("PERSONALITY_MODE")
    safe_t = _env_float("SAFE_ML_THRESHOLD")
    growth_t = _env_float("GROWTH_ML_THRESHOLD")

    vol_mult = _env_float("VOLUME_MULTIPLIER")
    sma_len = _env_int("SMA_LEN")
    min_adx = _env_float("MIN_ADX_FOR_BUY")
    min_atr_ratio = _env_float("MIN_ATR_RATIO_FOR_BUY")

    htf_trend = _env_bool("HTF_TREND_FOR_BUY", default=d.htf_trend_for_buy)
    htf_tf = _env_str("HTF_TF")
    htf_ema_len = _env_int("HTF_EMA_LEN")
    min_htf_bars = _env_int("MIN_HTF_BARS")

    cross_req = _env_bool("CROSS_REQUIRE_HTF_OR_VOLUME", default=d.cross_require_htf_or_volume)
    cross_vm = _env_float("CROSS_VOLUME_MULTIPLIER")
    cross_vs = _env_int("CROSS_VOLUME_SMA_LEN")

    ar = _env_str("ALLOWED_REGIMES")
    aer = _env_str("ALLOWED_ENTRY_REGIMES")
    allow_mr = _env_bool("ALLOW_MEAN_REVERSION_IN_RANGE", default=d.allow_mean_reversion_in_range)

    return GateConfig(
        ml_model_path=ml_path if ml_path is not None else d.ml_model_path,
        vol_model_path=vol_path if vol_path is not None else d.vol_model_path,
        vol_threshold=float(vol_th) if vol_th is not None else d.vol_threshold,
        personality_mode=personality if personality is not None else d.personality_mode,
        safe_threshold=float(safe_t) if safe_t is not None else float(d.safe_threshold),
        growth_threshold=float(growth_t) if growth_t is not None else float(d.growth_threshold),
        volume_multiplier=float(vol_mult) if vol_mult is not None else float(d.volume_multiplier),
        sma_len=int(sma_len) if sma_len is not None else int(d.sma_len),
        min_adx_for_buy=float(min_adx) if min_adx is not None else d.min_adx_for_buy,
        min_atr_ratio_for_buy=float(min_atr_ratio) if min_atr_ratio is not None else d.min_atr_ratio_for_buy,
        htf_trend_for_buy=bool(htf_trend),
        htf_tf=htf_tf if htf_tf is not None else d.htf_tf,
        htf_ema_len=int(htf_ema_len) if htf_ema_len is not None else int(d.htf_ema_len),
        min_htf_bars=int(min_htf_bars) if min_htf_bars is not None else int(d.min_htf_bars),
        cross_require_htf_or_volume=bool(cross_req),
        cross_volume_multiplier=float(cross_vm) if cross_vm is not None else float(d.cross_volume_multiplier),
        cross_volume_sma_len=int(cross_vs) if cross_vs is not None else int(d.cross_volume_sma_len),
        allowed_regimes=ar if ar is not None else d.allowed_regimes,
        allowed_entry_regimes=aer if aer is not None else d.allowed_entry_regimes,
        allow_mean_reversion_in_range=bool(allow_mr),
    )


def gate_kwargs_from_config(cfg: GateConfig) -> dict[str, Any]:
    """Convert GateConfig to the dict form used by run_spot_backtest / paper trading."""
    out: dict[str, Any] = {}
    if cfg.ml_model_path:
        out["ML_MODEL_PATH"] = str(cfg.ml_model_path)

    if cfg.vol_model_path:
        out["VOL_MODEL_PATH"] = str(cfg.vol_model_path)
    if cfg.vol_threshold is not None:
        out["VOL_THRESHOLD"] = float(cfg.vol_threshold)

    out["PERSONALITY_MODE"] = str(cfg.personality_mode)
    out["SAFE_ML_THRESHOLD"] = float(cfg.safe_threshold)
    out["GROWTH_ML_THRESHOLD"] = float(cfg.growth_threshold)

    out["VOLUME_MULTIPLIER"] = float(cfg.volume_multiplier)
    out["SMA_LEN"] = int(cfg.sma_len)
    if cfg.min_adx_for_buy is not None:
        out["MIN_ADX_FOR_BUY"] = float(cfg.min_adx_for_buy)
    if cfg.min_atr_ratio_for_buy is not None:
        out["MIN_ATR_RATIO_FOR_BUY"] = float(cfg.min_atr_ratio_for_buy)

    if cfg.htf_trend_for_buy:
        out["HTF_TREND_FOR_BUY"] = True
        if cfg.htf_tf is not None:
            out["HTF_TF"] = str(cfg.htf_tf)
        out["HTF_EMA_LEN"] = int(cfg.htf_ema_len)
        out["MIN_HTF_BARS"] = int(cfg.min_htf_bars)

    if cfg.cross_require_htf_or_volume:
        out["CROSS_REQUIRE_HTF_OR_VOLUME"] = True
        out["CROSS_VOLUME_MULTIPLIER"] = float(cfg.cross_volume_multiplier)
        out["CROSS_VOLUME_SMA_LEN"] = int(cfg.cross_volume_sma_len)

    if cfg.allowed_regimes is not None:
        out["ALLOWED_REGIMES"] = cfg.allowed_regimes

    if cfg.allowed_entry_regimes is not None:
        out["ALLOWED_ENTRY_REGIMES"] = cfg.allowed_entry_regimes
        out["ALLOW_MEAN_REVERSION_IN_RANGE"] = bool(cfg.allow_mean_reversion_in_range)

    return out
