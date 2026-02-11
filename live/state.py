from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class LivePosition:
    symbol: str
    amount: float
    entry: float
    stop: float
    target: float
    opened_at: str


class StateStore:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict[str, LivePosition]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        out: dict[str, LivePosition] = {}
        for sym, d in raw.items():
            out[sym] = LivePosition(
                symbol=str(d.get("symbol") or sym),
                amount=float(d.get("amount")),
                entry=float(d.get("entry")),
                stop=float(d.get("stop")),
                target=float(d.get("target")),
                opened_at=str(d.get("opened_at")),
            )
        return out

    def save(self, positions: dict[str, LivePosition]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {sym: asdict(pos) for sym, pos in positions.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
