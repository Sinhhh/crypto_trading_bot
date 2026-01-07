# crypto_trading

Source code lives under `crypto_trading/`.

**Quickstart**
- Install deps: `pip install -r requirements.txt`
- Fetch OHLCV from MEXC: `python3 scripts/mexc_fetch_ohlcv.py --timeframe 15m --symbols BTCUSDT,ETHUSDT`
- Run the CSV backtest: `python3 -m crypto_trading.backtest`
- Run the 15m template backtest: `python3 scripts/simulate_15m_template.py`

**Paper trading (SIM, live)**
- Runs a simulated wallet against live MEXC OHLCV (no real orders):
	- `python3 -m crypto_trading.simulator.paper_trading --symbol ETHUSDT --timeframe 1h --print-rows`
- Trades are appended to `reports/paper_trades_live_<SYMBOL>_<TF>_<UTC_TIMESTAMP>.csv`.

**Trade filters (optional, most important)**
- Avoid ranging markets: require ADX via `MIN_ADX_FOR_BUY` (backtest) or `--min-adx-for-buy` (paper)
- High volatility only: require ATR via `MIN_ATR_RATIO_FOR_BUY` (backtest) or `--min-atr-ratio-for-buy` (paper)
- HTF trend alignment (4H/Daily EMA): `HTF_TREND_FOR_BUY=True` (backtest) or `--htf-trend-for-buy` (paper, 15m bot only)
  - Backtest context: pass `context_files` with `{'4h': '...csv'}` and/or `{'1d': '...csv'}` and set `HTF_TF='4h'|'1d'`, `HTF_EMA_LEN=200`

**Data / outputs**
- Input CSVs are in `data/` (fetch script defaults to `data/raw/`).
- Generated trade reports are written to `reports/`.

**Secrets**
- Do not commit API keys.
- Set `MEXC_API_KEY` / `MEXC_SECRET_KEY` as environment variables if needed.
