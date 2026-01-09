# crypto_trading

Source code lives under `crypto_trading/`.

## Design Philosophy

This project is built around a few core principles:

- **Regime first**: Trading logic depends on market regime (trend, range, transition).
- **Default deny**: No trade is allowed unless explicitly permitted by regime and filters.
- **ML as a gate, not a predictor**: Machine learning is used to reject low-quality trades and scale risk, not to forecast price.
- **Multi-timeframe alignment**: Higher timeframes define context; lower timeframes execute.
- **Execution realism**: Backtests assume next-bar execution and include slippage.

The goal is not maximum trade count, but **robust selectivity** across market conditions.

## Quickstart

### 1) Setup

Create a virtualenv (recommended) and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2) Get data (CSV)

Fetch OHLCV CSVs (defaults to `data/raw/`). Examples:

```bash
# 1H data (recommended if you only trade 1H)
python3 -m scripts.fetch_mexc --timeframe 1h --symbols BTCUSDT,ETHUSDT

# 15m data (for the 15m MTF bot/template)
python3 -m scripts.fetch_mexc --timeframe 15m --symbols BTCUSDT,ETHUSDT --start-date 2025-12-01T00:00:00Z

# 4H data (used as context for 15m, and optional context for 1H)
python3 -m scripts.fetch_mexc --timeframe 4h --symbols BTCUSDT,ETHUSDT
```

Notes:
- `python3 -m ...` runs a module name, so do not include `.py`.
- If MEXC returns no candles (e.g. very old 15m history), the fetch script will error instead of writing a header-only CSV.

### 3) Run a backtest

The backtest is driven by CSVs in `data/raw/`.

#### 15m bot (MTF: entries on 15m, context from 1h + 4h)

Requires these files:
- `data/raw/BTCUSDT_15M.csv` (entry)
- `data/raw/BTCUSDT_1H.csv` (context)
- `data/raw/BTCUSDT_4H.csv` (context)

BTC example:

```bash
python3 -m scripts.backtest \
	--symbol BTCUSDT \
	--ohlcv-file data/raw/BTCUSDT_15M.csv \
	--timeframe 15M \
	--entry-tf 15m \
	--context-1h data/raw/BTCUSDT_1H.csv \
	--context-4h data/raw/BTCUSDT_4H.csv
```

ETH example:

```bash
python3 -m scripts.backtest \
	--symbol ETHUSDT \
	--ohlcv-file data/raw/ETHUSDT_15M.csv \
	--timeframe 15M \
	--entry-tf 15m \
	--context-1h data/raw/ETHUSDT_1H.csv \
	--context-4h data/raw/ETHUSDT_4H.csv
```

#### 1h bot (single timeframe, optional 4h context)

```bash
python3 -m scripts.backtest --symbol BTCUSDT --ohlcv-file data/raw/BTCUSDT_1H.csv --timeframe 1H --entry-tf 1h
python3 -m scripts.backtest --symbol ETHUSDT --ohlcv-file data/raw/ETHUSDT_1H.csv --timeframe 1H --entry-tf 1h
```

#### 4h bot (single timeframe)

```bash
python3 -m scripts.backtest --symbol BTCUSDT --ohlcv-file data/raw/BTCUSDT_4H.csv --timeframe 4H --entry-tf 4h
python3 -m scripts.backtest --symbol ETHUSDT --ohlcv-file data/raw/ETHUSDT_4H.csv --timeframe 4H --entry-tf 4h
```

Notes:
- The backtest auto-picks the first matching CSV under `data/raw/`.
- To switch symbol, pass `--symbol BTCUSDT` (or `--ohlcv-file path/to.csv`).
- If `data/raw/<SYMBOL>_4H.csv` exists, the 1H backtest will automatically load it as HTF context for optional filters.

**Paper trading (SIM, live)**
- Runs a simulated wallet against live MEXC OHLCV (no real orders):
	- `python3 -m scripts.paper_trade --symbol ETHUSDT --timeframe 1h --print-rows`
- Trades are appended to `reports/paper_trades_live_<SYMBOL>_<TF>_<UTC_TIMESTAMP>.csv`.

15m paper trading (uses live 15m entries, and pulls enough 1h/4h internally for MTF decisions):

```bash
python3 -m scripts.paper_trade --symbol BTCUSDT --timeframe 15m --entry-tf 15m --print-rows
python3 -m scripts.paper_trade --symbol ETHUSDT --timeframe 15m --entry-tf 15m --print-rows
```

## ML gate (optional)

This project supports an optional ML “gate” model that outputs a probability and is used to:
- reject low-quality BUY signals (entry gate)
- scale position size by confidence

It also supports an optional **volatility (should-trade) gate** model that answers:
"Is a big move likely within the next $N$ bars?".
When enabled, BUY entries are only allowed if `vol_prob >= VOL_THRESHOLD`.

### Train an ML model

Train ETH-only (recommended while iterating):

```bash
python3 -m scripts.train \
	--csv data/raw/ETHUSDT_1H.csv \
	--bar-minutes 60 \
	--horizon 24 \
	--calibration none \
	--sample-step 1 \
	--thresholds 0.50,0.55,0.60 \
	--out models/tp_sl_gate_eth_1h.joblib
```

Train BTC-only:

```bash
python3 -m scripts.train \
	--csv data/raw/BTCUSDT_1H.csv \
	--bar-minutes 60 \
	--horizon 24 \
	--calibration none \
	--sample-step 1 \
	--thresholds 0.50,0.55,0.60 \
	--out models/tp_sl_gate_btc_1h.joblib
```

Speed tips:
- `--sample-step 2` or `--sample-step 5` is much faster (fewer lifecycle simulations).
- `--calibration none` is fastest. Calibration (`sigmoid`/`isotonic`) can be slower and may compress probabilities.
- Use the printed `accept_rate` / `win_rate@accepted` to confirm the gate still accepts trades.

### Run the 1H backtest with ML enabled

Set the model path via env var and run the backtest:

```bash
ML_MODEL_PATH=models/tp_sl_gate_eth_1h.joblib python3 -m scripts.backtest
```

If ML fails to load, the backtest automatically falls back to the heuristic confidence.

### Run paper trading with ML enabled

Paper trading supports both:
- env var `ML_MODEL_PATH` (shared with backtest), and
- the explicit `--ml-model-path` flag (overrides env).

```bash
python3 -m scripts.paper_trade \
	--symbol ETHUSDT \
	--timeframe 1h \
	--ml-model-path models/tp_sl_gate_eth_1h.joblib \
	--print-rows
```

Note: paper trading also reads the same env-driven execution + gate knobs as the backtest (via `crypto_trading/config/config.py`). CLI flags override env when provided.

### Volatility (should-trade) gate

Train a volatility model (example: 15m, predict a 1% move within the next 24h):

```bash
python3 -m scripts.train_volatility \
	--csv data/raw/BTCUSDT_15M.csv \
	--bar-minutes 15 \
	--horizon 96 \
	--move-threshold 0.01 \
	--calibration sigmoid \
	--out models/vol_gate_btc_15m_24h_1pct.joblib
```

Enable the volatility gate in backtest (env vars):

```bash
VOL_MODEL_PATH=models/vol_gate_btc_15m_24h_1pct.joblib \
VOL_THRESHOLD=0.60 \
python3 -m scripts.backtest \
	--symbol BTCUSDT \
	--ohlcv-file data/raw/BTCUSDT_15M.csv \
	--timeframe 15M \
	--entry-tf 15m \
	--context-1h data/raw/BTCUSDT_1H.csv \
	--context-4h data/raw/BTCUSDT_4H.csv
```

Enable the volatility gate in paper trading (flags):

```bash
python3 -m scripts.paper_trade \
	--symbol BTCUSDT \
	--timeframe 15m \
	--entry-tf 15m \
	--vol-model-path models/vol_gate_btc_15m_24h_1pct.joblib \
	--vol-threshold 0.60 \
	--print-rows
```

**Trade filters (optional, most important)**
- Avoid ranging markets: require ADX via `MIN_ADX_FOR_BUY` (backtest) or `--min-adx-for-buy` (paper)
- High volatility only: require ATR via `MIN_ATR_RATIO_FOR_BUY` (backtest) or `--min-atr-ratio-for-buy` (paper)
- HTF trend alignment (4H/Daily EMA): `HTF_TREND_FOR_BUY=True` (backtest) or `--htf-trend-for-buy` (paper, 15m bot only)
  - Backtest context: pass `context_files` with `{'4h': '...csv'}` and/or `{'1d': '...csv'}` and set `HTF_TF='4h'|'1d'`, `HTF_EMA_LEN=200`

**CROSS_UP/CROSS_DOWN quality filter (recommended to reduce stop-out loops)**

Only allow `CROSS_UP` / `CROSS_DOWN` trades if confirmed by either:
- HTF regime alignment (when 4H context is available), OR
- extra-strong volume expansion.

Backtest (env vars):

```bash
# Enable the filter
CROSS_REQUIRE_HTF_OR_VOLUME=1 python3 -m scripts.backtest

# Optional tuning
CROSS_REQUIRE_HTF_OR_VOLUME=1 \
	CROSS_VOLUME_MULTIPLIER=1.7 \
	CROSS_VOLUME_SMA_LEN=20 \
	python3 -m scripts.backtest
```

Paper trading (flags):

```bash
python3 -m scripts.paper_trade \
	--symbol ETHUSDT \
	--timeframe 1h \
	--cross-require-htf-or-volume \
	--cross-volume-multiplier 1.7 \
	--cross-volume-sma-len 20 \
	--print-rows
```

**Explicit regime permission table (recommended for “default-deny” entries)**

This lets you explicitly decide which regimes are allowed to open new BUY entries.
Example policy:
- allow `TREND_UP`
- block `CROSS_UP` and `TRANSITION`
- block `RANGE` for trend entries (but still allow mean reversion if enabled)

Backtest (env vars):

```bash
ALLOWED_ENTRY_REGIMES='TREND_UP=1,RANGE=0,CROSS_UP=0,TRANSITION=0,TREND_DOWN=0,CROSS_DOWN=0' \
ALLOW_MEAN_REVERSION_IN_RANGE=1 \
python3 -m scripts.backtest
```

Paper trading (flags):

```bash
python3 -m scripts.paper_trade \
	--symbol ETHUSDT \
	--timeframe 1h \
	--allowed-entry-regimes 'TREND_UP=1,RANGE=0,CROSS_UP=0,TRANSITION=0,TREND_DOWN=0,CROSS_DOWN=0' \
	--allow-mean-reversion-in-range \
	--print-rows
```

## Other timeframes

The `unified_signal_mtf()` logic is for the 15m entry bot and requires 1H + 4H context.

### Train a 15m ML gate (BTC / ETH)

For a 24h horizon on 15m data: `horizon_bars = 24h / 15m = 96`.

```bash
# BTC 15m gate
python3 -m scripts.train \
	--csv data/raw/BTCUSDT_15M.csv \
	--bar-minutes 15 \
	--horizon 96 \
	--calibration sigmoid \
	--sample-step 1 \
	--out models/tp_sl_gate_btc_15m_24h.joblib

# ETH 15m gate
python3 -m scripts.train \
	--csv data/raw/ETHUSDT_15M.csv \
	--bar-minutes 15 \
	--horizon 96 \
	--calibration sigmoid \
	--sample-step 1 \
	--out models/tp_sl_gate_eth_15m_24h.joblib
```

Run a 15m backtest with the 15m ML gate:

```bash
ML_MODEL_PATH=models/tp_sl_gate_btc_15m_24h.joblib \
GROWTH_ML_THRESHOLD=0.55 \
python3 -m scripts.backtest \
	--symbol BTCUSDT \
	--ohlcv-file data/raw/BTCUSDT_15M.csv \
	--timeframe 15M \
	--entry-tf 15m \
	--context-1h data/raw/BTCUSDT_1H.csv \
	--context-4h data/raw/BTCUSDT_4H.csv
```

**Data / outputs**
- Input CSVs are in `data/` (fetch script defaults to `data/raw/`).
- Generated trade reports are written to `reports/`.

**Secrets**
- Do not commit API keys.
- Set `MEXC_API_KEY` / `MEXC_SECRET_KEY` as environment variables if needed.
