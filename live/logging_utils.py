import csv
import logging
import os
import sys
from datetime import datetime, timezone


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("live_trader")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s,%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def ensure_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ts_utc",
                "symbol",
                "event",
                "side",
                "entry",
                "stop",
                "target",
                "amount",
                "filled_price",
                "pnl_usd",
                "capital_usdt",
                "reason",
            ]
        )


def append_csv(path: str, row: dict) -> None:
    ensure_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                row.get("ts_utc"),
                row.get("symbol"),
                row.get("event"),
                row.get("side"),
                row.get("entry"),
                row.get("stop"),
                row.get("target"),
                row.get("amount"),
                row.get("filled_price"),
                row.get("pnl_usd"),
                row.get("capital_usdt"),
                row.get("reason"),
            ]
        )
