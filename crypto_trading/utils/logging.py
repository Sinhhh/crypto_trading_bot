from __future__ import annotations


def enable_sigpipe_default() -> None:
    """Avoid BrokenPipeError when piping output (e.g. `... | head`)."""
    try:  # pragma: no cover
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        return


def print_trade_row(
    time,
    symbol: str,
    side: str,
    pnl_usdt: float,
    *,
    cash: float | None = None,
    asset: float | None = None,
) -> None:
    if cash is None and asset is None:
        print(f"{time},{symbol},{side},{pnl_usdt:.2f}")
        return
    if asset is None:
        print(f"{time},{symbol},{side},{pnl_usdt:.2f},cash={cash:.2f}")
        return
    if cash is None:
        print(f"{time},{symbol},{side},{pnl_usdt:.2f},asset={asset:.8f}")
        return
    print(f"{time},{symbol},{side},{pnl_usdt:.2f},cash={cash:.2f},asset={asset:.8f}")
