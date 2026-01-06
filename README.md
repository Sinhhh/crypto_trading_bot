# crypto_trading

Source code lives under `src/crypto_trading/`.

**Quickstart**
- Install deps: `pip install -r requirements.txt`
- Run the 1H CSV backtest: `python3 bot.py`
- Run the 15m template backtest: `python3 scripts/simulate_15m_template.py`
- Fetch OHLCV from MEXC: `python3 scripts/mexc_fetch_ohlcv.py --timeframe 15m --symbols BTCUSDT,ETHUSDT`

**Data / outputs**
- Input CSVs are in `data/`.
- Generated trade reports are written to `reports/`.

**Secrets**
- Do not commit API keys.
- Set `MEXC_API_KEY` / `MEXC_SECRET_KEY` as environment variables if needed.
