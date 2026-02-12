# Crypto Intraday Bot (BTC & ETH) — Multi‑Timeframe SMC Framework

A deterministic, rule-based intraday trading system for **BTC** and **ETH** that follows the framework in [AGENTS.md](AGENTS.md):

- **4H**: market context → outputs **BUY / SELL / HOLD** bias only
- **1H**: setup validation → institutional footprint (BOS/CHOCH) + location (OB/FVG)
- **15M**: entry confirmation → liquidity sweep + candle confirmation + stop/target

This repo includes:
- A **CSV backtester** over provided OHLCV data
- A **paper trader** (spot-only) that pulls candles from Binance via `ccxt`, simulates fills, and logs to console + file
- A **live trader** (spot-only) that can place real orders (dry-run by default)

> Design constraints (by spec): BTC/ETH only, intraday holding (minutes to hours), only 4H/1H/15M timeframes, no indicator overload.

---

## What This Project Does

### Strategy pipeline (top-down)
1. **4H Bias** (`BUY` / `SELL` / `HOLD`)
   - Derived from 4H market structure only.
   - If structure is unclear/sideways → `HOLD` (no trades).

2. **1H Setup** (`setup_valid: true/false`)
   - Must align with the 4H bias.
   - Validates:
     - 1H market structure alignment
     - a footprint event: **BOS** or **CHOCH**
     - price location inside a 1H **Order Block (OB)** or **Fair Value Gap (FVG)**

3. **15M Entry** (`entry_signal: true/false`)
   - Must align with bias + setup.
   - Confirms:
     - **liquidity grab** (sweep + reclaim)
     - price in OB/FVG
     - candle confirmation (engulfing / pinbar / hammer or inside-bar breakout confirmation)
   - Computes:
     - `entry_price`
     - `stop` (structure-based, with optional ATR minimum distance)
     - `target` (RR-based)

The main “signal” output is produced by `generate_signal()` in [strategies/multi_timeframe.py](strategies/multi_timeframe.py).

---

## Repo Layout

- [strategies/](strategies/): strategy pipeline
- [indicators/](indicators/): structure, BOS/CHOCH, OB, FVG, liquidity grab detection
- [utils/](utils/): data/candle helpers
- [broker/](broker/): broker simulation + risk manager
- [exchange/](exchange/): ccxt exchange wrapper
- [trader/](trader/): paper trader implementation
- [backtest.py](backtest.py): CSV backtest runner + trade log output
- [logs/](logs/): per-run log files
- [data/raw/](data/raw/)
  - CSV candles for BTC/ETH at 15M/1H/4H

---

## Installation

### 1) Python environment
Recommended: Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Data
Backtest uses the provided CSVs under [data/raw/](data/raw/).

Paper trading fetches market data from Binance using `ccxt`.

---

## Backtesting (CSV)

Run:

```bash
python3 -m backtest
```

Options:

```bash
python3 -m backtest --symbols BTC --log-skips
```

Output:
- Console logs include `TRADE | ...` lines when trades trigger.
- Backtest results are written to logs/backtest_YYYYMMDD_HHMMSS.log (no CSV output).

Trade log format example:

```
YYYY-MM-DD HH:MM:SS | INFO | TRADE | BTC | SELL | Entry: ... | Exit: ... | PnL: ... | Capital: ...
```

---

## Paper Trading (Spot-only)

The paper trader:
- fetches 4H/1H/15M candles via Binance REST (`ccxt`)
- runs the same `generate_signal()` strategy logic
- simulates spot execution:
  - **BUY** may open a long
  - **SELL** may open a short (simulated)
- logs to console + a per-run log file

Run:

```bash
python3 -m trader.paper_trader --exchange binance --symbols BTC/USDT,ETH/USDT --poll 30
```

Useful options:

```bash
python3 -m trader.paper_trader --exchange binance --symbols BTC/USDT --poll 5 --cash 10000 --risk 0.01 \
  --log logs/paper.log
```

Console output:
- A line is printed every loop even if no trade is taken:

```
YYYY-MM-DD HH:MM:SS | INFO | SKIP | BTC | bias=HOLD | setup_valid=False | entry_signal=False | reason=...
```

- When a position opens/closes/exits, a `TRADE | ...` line is printed.

Paper log events include:
- `event`: `TRADE`, `SKIP`, `ERROR`
- `reason`: why the trade was skipped or how the trade exited

---

## Live Trading (Spot-only)

The live trader is under [live/](live/) and:
- fetches 4H/1H/15M candles via Binance REST (`ccxt`)
- runs the same `generate_signal()` strategy logic
- enforces spot-only execution (long-only)
- places **real market orders only if you pass `--live`**

Dry-run (recommended first; no API keys required):

```bash
python3 -m live.live_trader --symbols BTC/USDT,ETH/USDT --poll 30 --dry-capital 1000 --risk 0.01
```

Live trading (places real orders):

```bash
export BINANCE_API_KEY='...'
export BINANCE_API_SECRET='...'
python3 -m live.live_trader --live --symbols BTC/USDT,ETH/USDT --poll 30 --risk 0.01
```

Outputs:
- logs: [live/logs/live.log](live/logs/live.log)
- trade audit: [live/logs/live_trades.csv](live/logs/live_trades.csv)
- position state: [live/state.json](live/state.json)

---

## Risk Management

### Stop and target
Implemented in [strategies/multi_timeframe.py](strategies/multi_timeframe.py):
- Stop starts as structure-based (15M candle high/low).
- Optional ATR guard (enabled by default): enforce a minimum stop distance using **15M ATR(20) × 1.0**.
- Target is RR-based: `target = entry ± RR_MULT × |entry - stop|`.

### Paper sizing
Implemented in [broker/risk_manager.py](broker/risk_manager.py):
- Position size uses equity risk percent and stop distance.
- Also caps notional and applies a minimum notional guard.

---

## Determinism & Framework Compliance

This project is intentionally minimal:
- No extra timeframes beyond **4H/1H/15M**.
- No extra indicators like RSI/MACD/EMA/VWAP unless you explicitly add them.
- Signals are rule-based and deterministic.

---

## Known Limitations

- **Backtest** simulates stop/target per 15M candle; no fees/slippage.
- **Paper broker** is simplified: market orders fill with slippage/fees; no partial fills, no latency.

---

## Future Improvements (Safe & In-Scope)

These ideas keep the same 4H/1H/15M framework and remain deterministic:

1. **Execution realism**
   - Add fees + slippage to paper and backtest.
   - Add candle-by-candle stop/target simulation for backtests.

2. **Robustness & observability**
   - Add unit tests for structure/BOS/CHOCH, liquidity grab, and zone parsing.
   - Add a “health” log line with fetch durations and last candle timestamps.

3. **Risk & trade management**
   - Add partial take-profit + break-even logic after first target (still deterministic).
   - Add daily max loss / max trades per day guardrails.

4. **Data handling**
   - Cache fetched OHLCV to reduce API calls.
   - Validate candle continuity (missing bars) before generating signals.

---

## Quick Start

- Backtest:
  - `python3 -m backtest --symbols BTC,ETH`

- Paper trade (spot-only):
  - `python3 -m trader.paper_trader --exchange binance --symbols BTC/USDT,ETH/USDT --poll 30`

If you want, tell me whether you trade **spot only** or also want **perpetuals** later; I can keep the framework identical and only change the execution layer accordingly.
