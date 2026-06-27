from __future__ import annotations

import os
import time
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.parallel import effective_n_jobs
from sklearn.metrics import recall_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .backtest import run_backtest
from .labels import add_long_labels
from .modeling import feature_columns


GroupedIndices = list[tuple[str, np.ndarray]]


def valid_target_pairs(config: dict[str, Any]) -> list[tuple[float, float]]:
    opt = config["label_optimization"]
    return [
        (float(tp), float(sl))
        for tp, sl in product(opt["tp_candidates"], opt["sl_candidates"])
        if float(tp) > float(sl)
    ]


def outer_train_only(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cutoff = frame["open_time"].max() - pd.DateOffset(months=config["split"]["backtest_months"])
    research = frame[frame["open_time"] < cutoff]
    unique_times = np.array(sorted(research["open_time"].unique()))
    splitter = TimeSeriesSplit(n_splits=config["split"]["n_time_series_splits"])
    train_idx, _ = list(splitter.split(unique_times))[-1]
    train_times = set(unique_times[train_idx])
    return research[research["open_time"].isin(train_times)].copy()


def grouped_indices_by_symbol(sorted_frame: pd.DataFrame) -> GroupedIndices:
    return [
        (symbol, index.to_numpy())
        for symbol, index in sorted_frame.groupby("symbol", sort=False).groups.items()
    ]


def label_feature_frame_ordered(
    frame: pd.DataFrame,
    group_indices: GroupedIndices,
    config: dict[str, Any],
    tp_pct: float,
    sl_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.full(len(frame), np.nan, dtype=np.float32)
    forward_pnl = np.full(len(frame), np.nan, dtype=np.float32)
    exit_time = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")

    for _, positions in group_indices:
        symbol_frame = frame.iloc[positions]
        labeled = add_long_labels(symbol_frame, config, tp_pct=tp_pct, sl_pct=sl_pct)
        target[positions] = labeled["target"].to_numpy(dtype=np.float32)
        forward_pnl[positions] = labeled["forward_pnl"].to_numpy(dtype=np.float32)
        exit_time[positions] = labeled["exit_time"].to_numpy(dtype="datetime64[ns]")

    return target, forward_pnl, exit_time


def optimize_leveraged_threshold(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    min_recall = float(config["model"].get("min_recall_for_threshold", 0.05))
    best_threshold = 0.5
    best_report: dict[str, float] = {"leveraged_pnl_usdt": float("-inf"), "recall": 0.0}
    for threshold in np.arange(0.05, 0.96, 0.01):
        selected = probabilities >= threshold
        if selected.sum() == 0:
            continue
        recall = recall_score(validation["target"], selected.astype(int), zero_division=0)
        if recall < min_recall:
            continue
        _, report = run_backtest(
            validation,
            probabilities,
            float(threshold),
            config["backtest"]["fee_rate"],
            config["backtest"]["max_concurrent_positions"],
            config["backtest"]["fixed_margin_usdt"],
            config["backtest"]["leverage"],
            config["backtest"]["initial_balance_usdt"],
            config,
        )
        if report["leveraged_pnl_usdt"] > best_report["leveraged_pnl_usdt"]:
            best_threshold = float(threshold)
            best_report = {**report, "recall": float(recall)}
    return best_threshold, best_report


def evaluate_target_pair(
    feature_frame: pd.DataFrame,
    group_indices: GroupedIndices,
    config: dict[str, Any],
    tp_pct: float,
    sl_pct: float,
    xgb_threads: int,
) -> dict[str, float]:
    start = time.time()
    print(f"Evaluating TP: {tp_pct:.1%}, SL: {sl_pct:.1%}...", flush=True)
    target, forward_pnl, exit_time = label_feature_frame_ordered(feature_frame, group_indices, config, tp_pct, sl_pct)
    split_idx = int(len(feature_frame) * float(config["label_optimization"].get("train_fraction", 0.8)))
    split_idx = min(max(split_idx, 1), len(feature_frame) - 1)
    valid_exit = ~np.isnat(exit_time)
    train_end = split_idx
    val_start = split_idx
    while train_end > 1 and not valid_exit[train_end - 1]:
        train_end -= 1
    while val_start < len(feature_frame) - 1 and not valid_exit[val_start]:
        val_start += 1
    cols = feature_columns(feature_frame)
    y_train = target[:train_end]
    y_val = target[val_start:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        result = {
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "validation_pnl_usdt": float("-inf"),
            "threshold": 0.5,
            "recall": 0.0,
            "trades": 0,
            "duration_seconds": time.time() - start,
        }
        print(f"Evaluating TP: {tp_pct:.1%}, SL: {sl_pct:.1%}... skipped, single-class split", flush=True)
        return result

    params = dict(config["model"]["params"])
    params["n_estimators"] = int(config["label_optimization"].get("lightweight_estimators", 75))
    params["n_jobs"] = int(xgb_threads)
    params["verbosity"] = 0
    model = XGBClassifier(**params, random_state=config["model"]["random_state"])
    x_train = feature_frame[cols].iloc[:train_end]
    x_val = feature_frame[cols].iloc[val_start:]
    validation = feature_frame.iloc[val_start:].assign(
        target=target[val_start:],
        forward_pnl=forward_pnl[val_start:],
        exit_time=pd.to_datetime(exit_time[val_start:], utc=True),
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    probabilities = model.predict_proba(x_val)[:, 1]
    threshold, report = optimize_leveraged_threshold(validation, probabilities, config)
    result = {
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "validation_pnl_usdt": float(report["leveraged_pnl_usdt"]),
        "threshold": float(threshold),
        "recall": float(report.get("recall", 0.0)),
        "trades": int(report.get("total_trades", 0)),
        "duration_seconds": time.time() - start,
    }
    print(
        f"Evaluating TP: {tp_pct:.1%}, SL: {sl_pct:.1%}... "
        f"Expected Val PnL: {result['validation_pnl_usdt']:.2f} USDT, "
        f"Recall: {result['recall']:.2%}, Trades: {result['trades']}",
        flush=True,
    )
    return result


def optimize_label_targets(feature_frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    start = time.time()
    pairs = valid_target_pairs(config)
    if not pairs:
        raise ValueError("label_optimization produced no valid TP/SL pairs where TP > SL.")

    train_only = outer_train_only(feature_frame, config).sort_values(["open_time", "symbol"]).reset_index(drop=True)
    group_indices = grouped_indices_by_symbol(train_only)
    requested_jobs = min(4, max(1, (os.cpu_count() or 2) - 2))
    worker_count = max(1, effective_n_jobs(requested_jobs))
    total_cores = os.cpu_count() or worker_count
    xgb_threads = max(1, total_cores // worker_count)
    print(
        f"Starting TP/SL optimization: {len(pairs)} combinations, "
        f"n_jobs={requested_jobs}, workers={worker_count}, xgb_threads_per_worker={xgb_threads}",
        flush=True,
    )

    results = Parallel(n_jobs=requested_jobs)(
        delayed(evaluate_target_pair)(train_only, group_indices, config, tp, sl, xgb_threads)
        for tp, sl in pairs
    )
    ranked = sorted(results, key=lambda item: item["validation_pnl_usdt"], reverse=True)
    best = ranked[0]
    if best["validation_pnl_usdt"] == float("-inf"):
        raise RuntimeError("All TP/SL optimization candidates failed or produced invalid validation splits.")

    duration = time.time() - start
    print(
        f"TP/SL optimization completed in {duration:.1f}s. "
        f"Best TP: {best['tp_pct']:.1%}, SL: {best['sl_pct']:.1%}, "
        f"Expected Val PnL: {best['validation_pnl_usdt']:.2f} USDT",
        flush=True,
    )
    return {
        **best,
        "total_duration_seconds": duration,
        "evaluated_combinations": len(results),
    }
