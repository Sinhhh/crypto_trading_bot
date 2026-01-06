"""Script entrypoint for fetching OHLCV from MEXC.

Example:
  python3 scripts/mexc_fetch_ohlcv.py --timeframe 15m --symbols BTCUSDT,ETHUSDT
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from crypto_trading.io.mexc_fetch_ohlcv import main


if __name__ == "__main__":
    main()
