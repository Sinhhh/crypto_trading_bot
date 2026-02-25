class BacktestEngine:

    def __init__(
        self,
        broker,
        risk_manager,
        strategy,
        df_15m,
        df_1h,
        df_4h,
        lookback_15m,
        lookback_1h,
        lookback_4h,
        logger=None,
    ):
        self.broker = broker
        self.rm = risk_manager
        self.strategy = strategy

        self.df_15m = df_15m
        self.df_1h = df_1h
        self.df_4h = df_4h

        self.lb_15m = lookback_15m
        self.lb_1h = lookback_1h
        self.lb_4h = lookback_4h

        self.logger = logger

        self.equity_curve = []
        self.trade_log = []

    def run(self, symbol):

        for i in range(self.lb_15m, len(self.df_15m)):

            row = self.df_15m.iloc[i]
            ts = row["timestamp"]
            price = row["close"]

            # 1️⃣ Update market price
            self.broker.set_last_price(symbol, price)

            # 2️⃣ Check exits first
            closed = self.broker.check_stop_target(symbol)
            if closed:
                self.trade_log.append(closed)

            # 3️⃣ Record equity
            equity = self.broker.equity_usdt()
            self.equity_curve.append((ts, equity))

            # 4️⃣ Skip if in position
            if self.broker.get_position(symbol):
                continue

            # 5️⃣ Prepare lookbacks
            df_15m_lb = self.df_15m.iloc[i - self.lb_15m : i]
            df_1h_lb = self.df_1h[self.df_1h["timestamp"] <= ts].tail(self.lb_1h)
            df_4h_lb = self.df_4h[self.df_4h["timestamp"] <= ts].tail(self.lb_4h)

            # 6️⃣ Strategy signal
            signal = self.strategy.generate(df_4h_lb, df_1h_lb, df_15m_lb)

            if not signal["valid"]:
                continue

            # 7️⃣ Risk sizing
            size = self.rm.compute_size(
                symbol=symbol,
                side=signal["side"],
                entry=signal["entry"],
                stop=signal["stop"],
                target=signal["target"],
                setup_score=signal.get("score", 1.0),
            )

            if size <= 0:
                continue

            # 8️⃣ Execute order
            if signal["side"] == "LONG":
                result = self.broker.open_long(
                    symbol,
                    size,
                    signal["entry"],
                    signal["stop"],
                    signal["target"],
                )
            else:
                result = self.broker.open_short(
                    symbol,
                    size,
                    signal["entry"],
                    signal["stop"],
                    signal["target"],
                )

            if result["ok"] and self.logger:
                self.logger.info(f"OPEN {signal['side']} | size={size:.4f}")

        return self.equity_curve, self.trade_log
