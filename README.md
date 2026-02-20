# Crypto Intraday Bot (BTC & ETH)

## Multi-Timeframe SMC Framework — Deterministic & Rule-Based

A deterministic intraday trading system for BTC and ETH built on a strict Smart Money Concepts (SMC) top-down workflow.

This system follows the framework defined in [AGENTS.md](AGENTS.md):

- **4H**: market context → outputs **BUY / SELL / HOLD** bias only
- **1H**: setup validation → institutional footprint (BOS/CHOCH) + location (OB/FVG)
- **15M**: entry confirmation → liquidity sweep + candle confirmation + stop/target

## Strategy Architecture (Top-Down Workflow)

Below is the deterministic decision pipeline used for every trade:

```
                ┌─────────────────────────┐
                │        4H Context       │
                │  Market Structure Only  │
                │  → BUY / SELL / HOLD    │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │        1H Setup         │
                │  BOS / CHOCH + OB/FVG   │
                │  Alignment Required     │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │       15M Entry         │
                │ Liquidity Sweep +       │
                │ Candle Confirmation     │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │  Entry / Stop / Target  │
                │  RR-Based Execution     │
                └─────────────────────────┘
```

If any stage fails, no trade is placed. This prevents overtrading and enforces structural alignment across timeframes.

## Strategy Logic in Detail

1. **4H Bias — Market Context** (`BUY` / `SELL` / `HOLD`)
  - Derived from 4H market structure only.
  - Uses price-action filters (range compression + structure strength).
  - If structure is unclear/sideways → `HOLD` (no trades).

2. **1H Setup — Institutional Footprint**
  - Must align with the 4H bias.
  - Validates:
    - 1H market structure alignment
    - a footprint event: **BOS** or **CHOCH**
    - price location inside a 1H **Order Block (OB)** or **Fair Value Gap (FVG)**
    - HTF liquidity proximity (equal highs/lows)
  - Range-compression/low-volatility guard (price-action only)
  - Zone proximity filter to avoid far-away setups

3. **15M Entry — Confirmation Layer**
  - Must align with bias + setup.
  - Confirms:
    - liquidity sweep + reclaim of HTF liquidity
    - price in OB/FVG with first-tap validation
    - candle confirmation (engulfing, pinbar/hammer, or inside-bar breakout)
  - Computes:
    - `entry_price`
    - `stop` (15M structure with optional ATR guard)
    - `target` (RR-based)

The main signal output is produced by `generate_signal()` in [strategies/smc_signal.py](strategies/smc_signal.py). It returns:

- `bias_4h`, `setup_1h`, `entry_15m`
- Convenience fields: `bias`, `setup_valid`, `entry_signal`

## Repository Structure

```
strategies/        → Multi-timeframe pipeline
indicators/        → BOS, CHOCH, OB, FVG, liquidity logic
broker/            → Risk manager + execution simulation
exchange/          → ccxt exchange wrapper
trader/            → Paper trader
scripts/           → Data fetching helpers
utils/             → Candle helpers and shared utilities
backtest.py        → CSV backtest runner
main.py            → One-shot signal snapshot
logs/              → Runtime logs
data/raw/          → BTC & ETH OHLCV data (15M/1H/4H)
```

## Installation

### Python Environment

Recommended: Python 3.10+

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Backtesting uses CSV files in:

```
data/raw/
```

Paper trading fetches data via ccxt (default: Binance).

## Backtesting (CSV Engine)

Run:

```
python3 -m backtest
```

With options:

```
python3 -m backtest --symbols BTC --log-skips
```

Output:
- Console logs show `TRADE | ...` and `SKIP | ...`
- Log file saved to:

```
logs/backtest_YYYYMMDD_HHMMSS.log
```

Example:

```
YYYY-MM-DD HH:MM:SS | INFO | TRADE | BTC | SELL | Entry: ... | Exit: ... | PnL: ... | Capital: ...
```

## Paper Trading (Simulation)

Simulates:
- Candle fetching via REST (ccxt)
- Real-time signal generation
- Spot-like broker simulation (supports LONG and SHORT)
- Console + per-run file logging

Run:

```
python3 -m trader.paper_trader --exchange binance --symbols BTC/USDT,ETH/USDT --poll 30
```

Advanced example:

```
python3 -m trader.paper_trader \
  --exchange binance \
  --symbols BTC/USDT \
  --poll 5 \
  --cash 10000 \
  --risk 0.01 \
  --log logs/paper.log
```

Logs:
- Console `SKIP | ...` when conditions fail
- Console `TRADE | ...` when fills occur
- File logs are saved as `logs/paper_YYYYMMDD_HHMMSS.log`

## Optional Signal Snapshot

Run a one-shot signal evaluation with:

```
python3 -m main
```

By default it writes a CSV to `signals/output_signals.csv`. Create the `signals/` directory if you want that output.

## Risk Management

### Stop Logic
- Structure-based (latest 15M high/low)
- Optional ATR guard (see `utils/helpers.py`)

### Target Logic

`target = entry ± RR_MULT × |entry - stop|`

### Position Sizing

Defined in `broker/risk_manager.py` with fee and slippage guards.

## Known Limitations

### Backtest
- No fees
- No slippage
- Candle-level stop/target simulation

### Paper Trader
- No latency modeling
- No partial fills
- Simplified slippage and fee model

## Philosophy

This is not a signal toy. It is a strict execution engine built around:

- Multi-timeframe structural alignment
- Liquidity logic
- Institutional footprint detection
- Controlled risk exposure

If you later decide to trade perpetual futures instead of spot, the strategy layer remains identical and only the execution layer changes.
