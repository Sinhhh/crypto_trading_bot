import pandas as pd

from crypto_trading.regimes.regime_detector import detect_regime
from crypto_trading.strategies.trend_participation import trend_participation_signal
from crypto_trading.strategies.mean_reversion import mean_reversion_signal
from crypto_trading.strategies.ema_sma_crossover import ema_sma_cross_signal
from crypto_trading.strategies.breakout import breakout_signal


def unified_signal_with_meta(
    df_1h: pd.DataFrame,
    df_htf: pd.DataFrame | None = None,
    *,
    cross_require_htf_or_volume: bool = False,
    cross_volume_multiplier: float = 1.5,
    cross_volume_sma_len: int = 20,
) -> tuple[str, str, str]:
    """Unified signal with metadata.

    Returns: (signal, regime, source_strategy)
    """
    if df_1h is None or len(df_1h) < 60:
        return "HOLD", "TRANSITION", "none"

    regime = detect_regime(df_1h)

    def _cross_confirmed_by_htf(*, direction: str) -> bool:
        if df_htf is None or len(df_htf) < 220:
            return False
        try:
            htf_regime = str(detect_regime(df_htf)).upper()
        except Exception:
            return False
        if direction == "UP":
            return htf_regime in {"TREND_UP", "CROSS_UP"}
        if direction == "DOWN":
            return htf_regime in {"TREND_DOWN", "CROSS_DOWN"}
        return False

    def _cross_confirmed_by_volume() -> bool:
        if df_1h is None or "volume" not in df_1h.columns:
            return False
        if int(cross_volume_sma_len) <= 1:
            return False
        try:
            vol = df_1h["volume"].astype(float)
            avg = vol.rolling(int(cross_volume_sma_len)).mean().iloc[-1]
            last = float(vol.iloc[-1])
            if pd.isna(avg) or pd.isna(last):
                return False
            return last > float(avg) * float(cross_volume_multiplier)
        except Exception:
            return False

    # -------------------------
    # TREND REGIMES
    # -------------------------
    if regime in ("TREND_UP", "TREND_DOWN"):
        # 1. Trend pullback (primary)
        pullback = trend_participation_signal(
            df_1h,
            regime=regime,
        )
        if pullback != "HOLD":
            return pullback, regime, "trend_participation"

        # 2. Breakout continuation (secondary)
        breakout = breakout_signal(df_1h)
        if regime == "TREND_UP" and breakout == "BUY":
            return "BUY", regime, "breakout"
        if regime == "TREND_DOWN" and breakout == "SELL":
            return "SELL", regime, "breakout"

        return "HOLD", regime, "trend"

    # -------------------------
    # RANGE REGIME
    # -------------------------
    if regime == "RANGE":
        return (
            mean_reversion_signal(
                df_1h,
                regime=regime,
            ),
            regime,
            "mean_reversion",
        )

    # -------------------------
    # FRESH CROSSOVER REGIMES
    # -------------------------
    if regime == "CROSS_UP":
        if bool(cross_require_htf_or_volume):
            if not (
                _cross_confirmed_by_htf(direction="UP")
                or _cross_confirmed_by_volume()
            ):
                return "HOLD", regime, "ema_sma_crossover"
        signal = ema_sma_cross_signal(df_1h, min_separation=0.0005)
        return ("BUY" if signal == "BUY" else "HOLD"), regime, "ema_sma_crossover"

    if regime == "CROSS_DOWN":
        if bool(cross_require_htf_or_volume):
            if not (
                _cross_confirmed_by_htf(direction="DOWN")
                or _cross_confirmed_by_volume()
            ):
                return "HOLD", regime, "ema_sma_crossover"
        signal = ema_sma_cross_signal(df_1h, min_separation=0.0005)
        return ("SELL" if signal == "SELL" else "HOLD"), regime, "ema_sma_crossover"

    # -------------------------
    # TRANSITION / UNKNOWN
    # -------------------------
    return "HOLD", regime, "none"


def unified_signal_mtf(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> tuple[str, str, str]:
    """
    Multi-timeframe unified signal for a 15M entry bot.

    Returns: (signal, regime, source_strategy)

    Philosophy:
    - 4H decides regime (big picture)
    - 1H confirms permission / strength (mid picture)
    - 15M triggers the entry (execution timeframe)

    IMPORTANT:
    - Pass slices that include ONLY closed candles up to "now".
    - The backtest runner should compute:
        - signal using previous bar close
        - execute at next bar open
    """
    if df_15m is None or len(df_15m) < 120:
        return "HOLD", "TRANSITION", "none"
    if df_1h is None or len(df_1h) < 220:
        return "HOLD", "TRANSITION", "none"
    if df_4h is None or len(df_4h) < 220:
        return "HOLD", "TRANSITION", "none"

    # 1) Big picture regime from 4H
    regime_4h = detect_regime(df_4h)

    # 2) Mid picture confirmation from 1H
    regime_1h = detect_regime(df_1h)

    # Conservative permission logic:
    # - Prefer trading long only when 4H is not bearish.
    # - Use 1H to confirm momentum.
    if regime_4h in {"TREND_DOWN", "CROSS_DOWN"}:
        return "HOLD", regime_4h, "none"

    # -------------------------
    # TREND UP (preferred)
    # -------------------------
    if regime_4h == "TREND_UP":
        # Require 1H to not be bearish; ideally bullish
        if regime_1h in {"TREND_DOWN", "CROSS_DOWN"}:
            return "HOLD", regime_4h, "none"

        # 15M entry triggers:
        # 1) Trend pullback on 15M
        pullback = trend_participation_signal(df_15m, regime="TREND_UP")
        if pullback == "BUY":
            return "BUY", "TREND_UP", "trend_participation"

        # 2) Breakout continuation on 15M
        br = breakout_signal(df_15m)
        if br == "BUY":
            return "BUY", "TREND_UP", "breakout"

        return "HOLD", "TREND_UP", "trend"

    # -------------------------
    # RANGE (optional)
    # -------------------------
    if regime_4h == "RANGE":
        # Optional: allow mean reversion only if 1H also range/transition (avoid catching knives)
        if regime_1h in {"TREND_DOWN"}:
            return "HOLD", "RANGE", "none"

        sig = mean_reversion_signal(df_15m, regime="RANGE")
        return (sig, "RANGE", "mean_reversion")

    # -------------------------
    # FRESH CROSS UP (early trend)
    # -------------------------
    if regime_4h == "CROSS_UP":
        # If 1H already confirms trend up, treat like trend
        if regime_1h == "TREND_UP":
            pullback = trend_participation_signal(df_15m, regime="TREND_UP")
            if pullback == "BUY":
                return "BUY", "CROSS_UP", "trend_participation"
            br = breakout_signal(df_15m)
            if br == "BUY":
                return "BUY", "CROSS_UP", "breakout"
            return "HOLD", "CROSS_UP", "trend"

        # Otherwise, allow only stronger crossover signals on 15M
        cross = ema_sma_cross_signal(df_15m, min_separation=0.0005)
        if cross == "BUY":
            return "BUY", "CROSS_UP", "ema_sma_crossover"

        return "HOLD", "CROSS_UP", "ema_sma_crossover"

    # -------------------------
    # TRANSITION / UNKNOWN
    # -------------------------
    # In transition, be conservative: only allow breakout with strong momentum (your gating handles volume etc.)
    br = breakout_signal(df_15m)
    if br == "BUY" and regime_1h in {"CROSS_UP", "TREND_UP"}:
        return "BUY", "TRANSITION", "breakout"

    return "HOLD", "TRANSITION", "none"
