from __future__ import annotations

import numpy as np
import pandas as pd


def _scan_path(
    path: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    horizon_end: pd.Timestamp,
    tp_pct: float,
    sl_pct: float,
) -> tuple[int, float, pd.Timestamp | pd.NaT]:
    future = path[(path["open_time"] > entry_time) & (path["open_time"] <= horizon_end)].sort_values("open_time")
    if future.empty:
        return 0, 0.0, pd.NaT

    tp = entry_price * (1 + tp_pct)
    sl = entry_price * (1 - sl_pct)
    last_close = float(future["close"].iloc[-1])
    last_time = future["open_time"].iloc[-1]

    for row in future.itertuples(index=False):
        hit_sl = row.low <= sl
        hit_tp = row.high >= tp
        if hit_sl and hit_tp:
            return 0, -sl_pct, row.open_time
        if hit_sl:
            return 0, -sl_pct, row.open_time
        if hit_tp:
            return 1, tp_pct, row.open_time
    return 0, (last_close - entry_price) / entry_price, last_time


def add_long_labels(
    frame: pd.DataFrame,
    config: dict,
    confirmation_frame_15m: pd.DataFrame | None = None,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
) -> pd.DataFrame:
    tp_pct = float(tp_pct if tp_pct is not None else config["label"]["take_profit_pct"])
    sl_pct = float(sl_pct if sl_pct is not None else config["label"]["stop_loss_pct"])
    horizon = config["label"]["max_holding_bars_5m"]
    df = frame.sort_values("open_time").reset_index(drop=True).copy()

    labels = np.zeros(len(df), dtype=int)
    pnl = np.zeros(len(df), dtype=float)
    exit_times: list[pd.Timestamp | pd.NaT] = [pd.NaT] * len(df)
    labels_15m = np.full(len(df), np.nan)
    pnl_15m = np.full(len(df), np.nan)

    closes = df["close"].to_numpy()
    times = df["open_time"].to_list()
    path_5m = df[["open_time", "high", "low", "close"]].copy()
    path_15m = None
    if confirmation_frame_15m is not None:
        path_15m = confirmation_frame_15m[["open_time", "high", "low", "close"]].copy()

    for i, entry in enumerate(closes):
        end = min(i + horizon, len(df) - 1)
        horizon_end = times[end]
        outcome, trade_pnl, exit_time = _scan_path(path_5m, times[i], float(entry), horizon_end, tp_pct, sl_pct)

        labels[i] = outcome
        pnl[i] = trade_pnl
        exit_times[i] = exit_time

        if path_15m is not None:
            outcome_15m, trade_pnl_15m, _ = _scan_path(path_15m, times[i], float(entry), horizon_end, tp_pct, sl_pct)
            labels_15m[i] = outcome_15m
            pnl_15m[i] = trade_pnl_15m

    df["target"] = labels
    df["forward_pnl"] = pnl
    df["exit_time"] = exit_times
    if path_15m is not None:
        df["target_15m_check"] = labels_15m
        df["forward_pnl_15m_check"] = pnl_15m
    return df
