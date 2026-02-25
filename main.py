"""
Entry point for Multi-Timeframe Intraday BTC & ETH Trading Signals
Folder structure:
- data/                        -> OHLCV CSV files
- src/core  -> Structure, OB, FVG, Liquidity, strategies
- src/data  -> Data loading utilities
- src/utils -> Candle utils, helpers
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from data.data_loader import load_csv_data
from core.strategies.smc_signal import generate_signal


# ===========================
# CONFIGURATION
# ===========================
SYMBOLS = ["BTC", "ETH"]
TIMEFRAMES = ["4H", "1H", "15M"]
DATA_PATH = "data/raw/"
RESULT_PATH = "signals/output_signals.csv"  # Optional CSV output


# ===========================
# MAIN SIGNAL LOOP
# ===========================
def main():
    all_signals = []

    for symbol in SYMBOLS:
        # Load OHLCV data for 4H, 1H, 15M
        df_4h = load_csv_data(f"{DATA_PATH}{symbol}_4H.csv")
        df_1h = load_csv_data(f"{DATA_PATH}{symbol}_1H.csv")
        df_15m = load_csv_data(f"{DATA_PATH}{symbol}_15M.csv")

        # Ensure enough data
        if len(df_4h) < 5 or len(df_1h) < 5 or len(df_15m) < 5:
            print(f"Not enough data for {symbol}")
            continue

        # Generate signals using multi-timeframe strategy
        signals = generate_signal(df_4h, df_1h, df_15m)

        # Add metadata
        signals["symbol"] = symbol
        signals["timestamp"] = pd.Timestamp.now()

        all_signals.append(signals)

        # Print to console
        print(
            f"{symbol} | Bias: {signals['bias']} | Setup Valid: {signals['setup_valid']} | Entry Signal: {signals['entry_signal']}"
        )

    # Optionally save to CSV
    if all_signals:
        pd.DataFrame(all_signals).to_csv(RESULT_PATH, index=False)
        print(f"Signals saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
