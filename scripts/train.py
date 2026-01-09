from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

# Model: calibrated logistic regression on tabular features
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crypto_trading.io.loader import load_ohlcv
from crypto_trading.ml.tp_sl_dataset import TpSlDatasetConfig, build_tp_before_sl_dataset


def _configure_logging(level: str) -> None:
    lvl = str(level or "INFO").strip().upper()
    numeric = getattr(logging, lvl, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _time_split(df: pd.DataFrame, *, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TP-before-SL ML gate model")
    ap.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to OHLCV CSV (repeat for multiple symbols)",
    )
    ap.add_argument(
        "--out",
        default="models/tp_sl_gate_1h_24h.joblib",
        help="Output joblib path",
    )

    ap.add_argument("--horizon", type=int, default=24, help="Horizon bars")
    ap.add_argument(
        "--bar-minutes",
        type=int,
        default=60,
        help="Bar duration in minutes (e.g. 60=1h, 15=15m, 5=5m)",
    )
    ap.add_argument(
        "--atr-mult",
        type=float,
        default=2.2,
        help="Base ATR stop multiplier (used when volatility estimate is unavailable)",
    )
    ap.add_argument(
        "--unresolved",
        choices=["drop", "zero"],
        default="drop",
        help="How to handle unresolved samples",
    )

    ap.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Speed knob: only evaluate every Nth candidate bar when building the dataset (1=all)",
    )

    ap.add_argument(
        "--calibration",
        choices=["none", "sigmoid", "isotonic"],
        default="isotonic",
        help="Calibration method. 'none' is fastest; 'sigmoid' is usually a good fast default.",
    )
    ap.add_argument(
        "--calib-splits",
        type=int,
        default=3,
        help="TimeSeriesSplit folds for calibration (ignored if --calibration=none)",
    )

    ap.add_argument(
        "--thresholds",
        default="0.55,0.80",
        help="Comma-separated probability thresholds to report (e.g. '0.55,0.80')",
    )

    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING)",
    )

    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)

    args = ap.parse_args()

    _configure_logging(args.log_level)
    log = logging.getLogger("scripts.train")
    t0 = time.perf_counter()
    log.info(
        "Starting training: horizon=%s bar_minutes=%s sample_step=%s calib=%s calib_splits=%s",
        int(args.horizon),
        int(args.bar_minutes),
        int(args.sample_step),
        str(args.calibration),
        (
            "n/a"
            if str(args.calibration).strip().lower() == "none"
            else int(args.calib_splits)
        ),
    )

    csv_paths = [str(x) for x in (args.csv or []) if str(x).strip()]
    all_ds = []

    for csv_path in csv_paths:
        log.info("Loading OHLCV: %s", csv_path)
        t_load0 = time.perf_counter()
        df = load_ohlcv(csv_path)
        t_load1 = time.perf_counter()
        try:
            idx0 = df.index.min()
            idx1 = df.index.max()
        except Exception:
            idx0, idx1 = None, None
        log.info(
            "Loaded %s rows in %.2fs (range=%s..%s)",
            len(df),
            (t_load1 - t_load0),
            idx0,
            idx1,
        )

        # Infer symbol from filename like BTCUSDT_1H.csv
        stem = Path(csv_path).stem
        inferred_symbol = stem.split("_")[0] if "_" in stem else stem

        cfg = TpSlDatasetConfig(
            horizon_bars=int(args.horizon),
            bar_minutes=int(args.bar_minutes),
            lifecycle_atr_multiplier=float(args.atr_mult),
            unresolved=str(args.unresolved),
            sample_step=int(args.sample_step),
        )

        log.info("Building dataset for symbol=%s", inferred_symbol)
        t_ds0 = time.perf_counter()
        ds = build_tp_before_sl_dataset(df, cfg=cfg, symbol=inferred_symbol)
        t_ds1 = time.perf_counter()
        if ds is None or ds.empty:
            log.warning("No samples produced from %s", csv_path)
            continue
        log.info(
            "Dataset built: n=%s pos_rate=%.3f in %.2fs",
            len(ds),
            float(ds["y"].mean()) if len(ds) else float("nan"),
            (t_ds1 - t_ds0),
        )
        all_ds.append(ds)

    if not all_ds:
        raise SystemExit("No training samples produced from any CSV.")

    ds = pd.concat(all_ds, axis=0, ignore_index=True)
    log.info("Merged dataset: total=%s pos_rate=%.3f", len(ds), float(ds["y"].mean()))

    ds = ds.sort_values("time")

    train_df, val_df, test_df = _time_split(
        ds, train_frac=float(args.train_frac), val_frac=float(args.val_frac)
    )
    log.info("Split sizes: train=%s val=%s test=%s", len(train_df), len(val_df), len(test_df))

    feature_cols = [
        "symbol",
        "bar_of_day",
        "day_of_week",
        "bars_to_utc_midnight",
        "close",
        "volume",
        "ret_1",
        "ret_3",
        "ret_6",
        "rv_20",
        "rsi14",
        "adx14",
        "dist_sma",
        "vol_ratio",
        "regime",
        "source",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["y"].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df["y"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["y"].astype(int)

    num_cols = [
        "bar_of_day",
        "day_of_week",
        "bars_to_utc_midnight",
        "close",
        "volume",
        "ret_1",
        "ret_3",
        "ret_6",
        "rv_20",
        "rsi14",
        "adx14",
        "dist_sma",
        "vol_ratio",
    ]
    cat_cols = ["symbol", "regime", "source"]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    base = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", base)])

    calib = str(args.calibration).strip().lower()
    if calib != "none":
        tscv = TimeSeriesSplit(n_splits=int(args.calib_splits))
        pipe = CalibratedClassifierCV(
            pipe,
            method=calib,
            cv=tscv,
        )

    log.info("Fitting model...")
    pipe.fit(X_train, y_train)

    def _eval(split_name: str, X, y):
        proba = pipe.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
        brier = brier_score_loss(y, proba)
        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        log.info(
            "%s: auc=%.4f brier=%.4f acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
            split_name,
            float(auc),
            float(brier),
            float(acc),
            float(prec),
            float(rec),
            float(f1),
        )
        return proba

    _ = _eval("val", X_val, y_val)
    proba_test = _eval("test", X_test, y_test)

    # Threshold report
    thresholds = []
    for part in str(args.thresholds).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            thresholds.append(float(part))
        except Exception:
            continue

    if thresholds:
        for th in thresholds:
            accepted = proba_test >= float(th)
            accept_rate = float(accepted.mean()) if len(accepted) else float("nan")
            if accepted.any():
                win_rate = float(y_test[accepted].mean())
            else:
                win_rate = float("nan")
            log.info(
                "threshold=%.2f accept_rate=%.3f win_rate@accepted=%.3f",
                float(th),
                float(accept_rate),
                float(win_rate),
            )

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, out_path)
    log.info("Saved model: %s", out_path)
    log.info("Done in %.2fs", (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
