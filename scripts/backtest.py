from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal


# Support running as a plain script: `python3 scripts/backtest.py ...`
# (Recommended usage remains: `python3 -m scripts.backtest ...`)
if __package__ is None:  # pragma: no cover
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _infer_entry_tf_from_label(timeframe: str) -> Literal["5m", "15m", "1h", "4h"]:
    tf_lower = str(timeframe).lower().replace("min", "m")
    if "5" in tf_lower and "15" not in tf_lower:
        return "5m"
    if "15" in tf_lower:
        return "15m"
    if "4h" in tf_lower or "240" in tf_lower:
        return "4h"
    return "1h"


def _infer_timeframe_from_filename(symbol: str, p: Path) -> str:
    stem = p.stem
    pref = f"{symbol}_"
    if stem.startswith(pref):
        return stem[len(pref) :]
    return "UNKNOWN"


def _pick_ohlcv_file(*, data_dir: Path, symbol: str) -> Path | None:
    preferred = [
        data_dir / f"{symbol}_5M.csv",
        data_dir / f"{symbol}_15M.csv",
        data_dir / f"{symbol}_1H.csv",
        data_dir / f"{symbol}_4H.csv",
    ]
    p = next((x for x in preferred if x.exists()), None)
    if p is not None:
        return p
    if not data_dir.exists():
        return None
    candidates = sorted(data_dir.glob(f"{symbol}_*.csv"))
    return candidates[0] if candidates else None


def main() -> None:
    p = argparse.ArgumentParser(description="Run spot backtest")
    p.add_argument("--symbol", default="BTCUSDT", help="Symbol like BTCUSDT")
    p.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing OHLCV CSVs (default: data/raw)",
    )
    p.add_argument(
        "--ohlcv-file",
        default=None,
        help="Explicit OHLCV CSV path (overrides auto-pick)",
    )
    p.add_argument(
        "--timeframe",
        default=None,
        help="Label for printing/inference (default inferred from filename)",
    )
    p.add_argument(
        "--entry-tf",
        default=None,
        choices=["5m", "15m", "1h", "4h"],
        help="Bot entry timeframe (default inferred from timeframe)",
    )
    p.add_argument("--starting-balance", type=float, default=100.0)
    p.add_argument("--max-candles", type=int, default=25000)
    p.add_argument("--lookback-bars", type=int, default=None)
    p.add_argument("--bar-step", type=int, default=1)
    p.add_argument("--print-rows", action="store_true", help="Print trade rows")
    p.add_argument(
        "--context-1h",
        default=None,
        help="Optional 1H context CSV (for 15m bot)",
    )
    p.add_argument(
        "--context-4h",
        default=None,
        help="Optional 4H context CSV (recommended for CROSS confirmation)",
    )
    p.add_argument(
        "--context-1d",
        default=None,
        help="Optional 1D context CSV (for HTF trend gating)",
    )

    args = p.parse_args()

    # Allow piping output without crashing (e.g. `... | head`).
    try:  # pragma: no cover
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass

    symbol = str(args.symbol).strip().upper()
    data_dir = Path(str(args.data_dir))

    if args.ohlcv_file:
        ohlcv_path = Path(str(args.ohlcv_file))
    else:
        ohlcv_path = _pick_ohlcv_file(data_dir=data_dir, symbol=symbol)

    if ohlcv_path is None or (not ohlcv_path.exists()):
        raise SystemExit(f"OHLCV CSV not found (symbol={symbol}) under: {data_dir}")

    timeframe = str(args.timeframe).strip() if args.timeframe else _infer_timeframe_from_filename(symbol, ohlcv_path)
    entry_tf: Literal["5m", "15m", "1h", "4h"] = (
        str(args.entry_tf).strip().lower() if args.entry_tf else _infer_entry_tf_from_label(timeframe)
    )  # type: ignore[assignment]

    # Context files: explicit flags override auto-pick.
    context_files: dict[str, str] | None = None
    if args.context_1h or args.context_4h or args.context_1d:
        context_files = {}
        if args.context_1h:
            context_files["1h"] = str(Path(str(args.context_1h)))
        if args.context_4h:
            context_files["4h"] = str(Path(str(args.context_4h)))
        if args.context_1d:
            context_files["1d"] = str(Path(str(args.context_1d)))
    else:
        # Match the previous default behavior.
        if entry_tf == "15m":
            ctx_1h = data_dir / f"{symbol}_1H.csv"
            ctx_4h = data_dir / f"{symbol}_4H.csv"
            if ctx_1h.exists() and ctx_4h.exists():
                context_files = {"1h": str(ctx_1h), "4h": str(ctx_4h)}
            else:
                # Fallback: run the single-timeframe bot.
                entry_tf = "1h"

        if context_files is None and entry_tf == "1h":
            ctx_4h = data_dir / f"{symbol}_4H.csv"
            if ctx_4h.exists():
                context_files = {"4h": str(ctx_4h)}

    from crypto_trading.backtest.spot_backtest import run_spot_backtest
    from crypto_trading.config.config import (
        gate_kwargs_from_config,
        load_execution_config_from_env,
        load_gate_config_from_env,
    )

    exec_cfg = load_execution_config_from_env()
    gate_cfg = load_gate_config_from_env()
    gate_kwargs = gate_kwargs_from_config(gate_cfg)

    run_spot_backtest(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv_file=str(ohlcv_path),
        starting_balance=float(args.starting_balance),
        max_candles=int(args.max_candles) if args.max_candles is not None else None,
        lookback_bars=int(args.lookback_bars) if args.lookback_bars is not None else None,
        bar_step=int(args.bar_step),
        entry_tf=entry_tf,
        context_files=context_files,
        gate_kwargs=gate_kwargs if gate_kwargs else None,
        theoretical_max=bool(exec_cfg.theoretical_max),
        slippage_rate=float(exec_cfg.slippage_rate),
        fee_rate=float(exec_cfg.fee_rate),
        print_rows=bool(args.print_rows),
    )


if __name__ == "__main__":
    main()
