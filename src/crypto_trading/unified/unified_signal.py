import pandas as pd

from crypto_trading.regimes.regime_detector import detect_regime
from crypto_trading.strategies.trend_participation import trend_participation_signal
from crypto_trading.strategies.mean_reversion import mean_reversion_signal
from crypto_trading.strategies.ema_sma_crossover import ema_sma_cross_signal
from crypto_trading.strategies.breakout import breakout_signal


def unified_signal(df_1h: pd.DataFrame) -> str:
    """
    Unified regime-aware signal.

    Returns only: "BUY" / "SELL" / "HOLD".
    """
    signal, _regime, _source = unified_signal_with_meta(df_1h)
    return signal


def unified_signal_with_meta(df_1h: pd.DataFrame) -> tuple[str, str, str]:
    """Unified signal with metadata.

    Returns: (signal, regime, source_strategy)
    """
    if df_1h is None or len(df_1h) < 60:
        return "HOLD", "TRANSITION", "none"

    regime = detect_regime(df_1h)

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
        signal = ema_sma_cross_signal(df_1h)
        return ("BUY" if signal == "BUY" else "HOLD"), regime, "ema_sma_crossover"

    if regime == "CROSS_DOWN":
        signal = ema_sma_cross_signal(df_1h)
        return ("SELL" if signal == "SELL" else "HOLD"), regime, "ema_sma_crossover"

    # -------------------------
    # TRANSITION / UNKNOWN
    # -------------------------
    return "HOLD", regime, "none"
