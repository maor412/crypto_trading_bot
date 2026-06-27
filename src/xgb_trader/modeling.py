from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


NON_FEATURE_COLUMNS = {
    "symbol",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "target",
    "target_15m_check",
    "forward_pnl",
    "forward_pnl_15m_check",
    "exit_time",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])]


def chronological_split(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff = frame["open_time"].max() - pd.DateOffset(months=config["split"]["backtest_months"])
    research = frame[frame["open_time"] < cutoff].sort_values("open_time")
    backtest = frame[frame["open_time"] >= cutoff].sort_values("open_time")
    splitter = TimeSeriesSplit(n_splits=config["split"]["n_time_series_splits"])
    unique_times = np.array(sorted(research["open_time"].unique()))
    train_idx, test_idx = list(splitter.split(unique_times))[-1]
    train_times = set(unique_times[train_idx])
    test_times = set(unique_times[test_idx])
    train = research[research["open_time"].isin(train_times)]
    test = research[research["open_time"].isin(test_times)]
    return train.copy(), test.copy(), backtest.copy()


def train_model(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> tuple[XGBClassifier, list[str], np.ndarray]:
    cols = feature_columns(train)
    model = XGBClassifier(**config["model"]["params"], random_state=config["model"]["random_state"])
    model.fit(
        train[cols],
        train["target"],
        eval_set=[(test[cols], test["target"])],
        verbose=False,
    )
    probabilities = model.predict_proba(test[cols])[:, 1]
    return model, cols, probabilities


def threshold_metrics(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    fee_rate: float,
) -> dict[str, float]:
    selected = probabilities >= threshold
    trades = int(selected.sum())
    preds = selected.astype(int)
    precision = precision_score(test["target"], preds, zero_division=0)
    recall = recall_score(test["target"], preds, zero_division=0)
    if trades == 0:
        profit = 0.0
        profit_factor = 0.0
    else:
        trade_returns = test.loc[selected, "forward_pnl"].to_numpy() - (2 * fee_rate)
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        profit = float(trade_returns.sum())
    return {
        "threshold": float(threshold),
        "profit": profit,
        "precision": float(precision),
        "recall": float(recall),
        "profit_factor": float(profit_factor),
        "trades": trades,
    }


def optimize_threshold(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    fee_rate: float,
    min_recall: float = 0.05,
    config: dict | None = None,
) -> dict[str, float]:
    threshold_config = (config or {}).get("model", {}).get("threshold_optimization", {})
    min_precision = float(threshold_config.get("min_precision", 0.55))
    min_trades = int(threshold_config.get("min_trades", 50))
    min_profit_factor = float(threshold_config.get("min_profit_factor", 1.0))
    fallback_threshold = float(threshold_config.get("fallback_threshold", 0.70))

    best_valid: dict[str, float] = {
        "threshold": 0.5,
        "profit": float("-inf"),
        "precision": 0.0,
        "recall": 0.0,
        "profit_factor": 0.0,
        "trades": 0,
        "selection_reason": "unset",
    }
    best_precision: dict[str, float] | None = None

    for threshold in np.arange(0.05, 0.96, 0.01):
        metrics = {
            **threshold_metrics(test, probabilities, float(threshold), fee_rate),
            "min_precision": min_precision,
            "min_trades": min_trades,
            "min_profit_factor": min_profit_factor,
            "min_recall": float(min_recall),
        }
        if metrics["trades"] == 0:
            continue

        if (
            best_precision is None
            or metrics["precision"] > best_precision["precision"]
            or (
                metrics["precision"] == best_precision["precision"]
                and metrics["trades"] > best_precision["trades"]
            )
        ):
            best_precision = metrics

        valid = (
            metrics["precision"] >= min_precision
            and metrics["trades"] >= min_trades
            and metrics["profit_factor"] > min_profit_factor
        )
        if not valid:
            continue

        if (
            best_valid["selection_reason"] == "unset"
            or metrics["profit_factor"] > best_valid["profit_factor"]
            or (
                metrics["profit_factor"] == best_valid["profit_factor"]
                and metrics["precision"] > best_valid["precision"]
            )
        ):
            best_valid = {**metrics, "selection_reason": "max_profit_factor_with_sniper_constraints"}

    if best_valid["selection_reason"] != "unset":
        return best_valid

    if best_precision is not None:
        threshold = max(float(best_precision["threshold"]), fallback_threshold)
        metrics = threshold_metrics(test, probabilities, threshold, fee_rate)
        return {
            **metrics,
            "selection_reason": "fallback_highest_precision_hardened_to_min_threshold",
            "fallback_threshold": fallback_threshold,
            "best_precision_threshold": float(best_precision["threshold"]),
            "best_precision": float(best_precision["precision"]),
            "min_precision": min_precision,
            "min_trades": min_trades,
            "min_profit_factor": min_profit_factor,
            "min_recall": float(min_recall),
        }

    return {
        "threshold": fallback_threshold,
        "profit": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "profit_factor": 0.0,
        "trades": 0,
        "selection_reason": "fallback_hardcoded_high_threshold",
        "fallback_threshold": fallback_threshold,
        "min_precision": min_precision,
        "min_trades": min_trades,
        "min_profit_factor": min_profit_factor,
        "min_recall": float(min_recall),
    }


def save_artifacts(model: XGBClassifier, columns: list[str], threshold_metrics: dict[str, float], config: dict) -> None:
    output_dir = Path(config["model"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "xgb_model.joblib")
    joblib.dump(columns, output_dir / "feature_columns.joblib")
    joblib.dump(threshold_metrics, output_dir / "threshold.joblib")
