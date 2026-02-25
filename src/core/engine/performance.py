import pandas as pd
import numpy as np


def analyze_equity(equity_curve):

    df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
    df["returns"] = df["equity"].pct_change().fillna(0)

    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1

    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    sharpe = (
        np.mean(df["returns"]) / np.std(df["returns"]) * np.sqrt(365 * 24 * 4)
        if np.std(df["returns"]) > 0
        else 0
    )

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }
