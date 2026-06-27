from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed

from .backtest import run_backtest
from .binance_client import BinanceFuturesClient
from .config import ensure_dirs, load_config
from .data import download_market_data, load_cached_market_data
from .features import build_symbol_dataset
from .labels import add_long_labels
from .modeling import chronological_split, feature_columns, optimize_threshold, save_artifacts, train_model
from .optimize_targets import optimize_label_targets


def processing_jobs(config: dict) -> int:
    return max(1, int(config.get("processing", {}).get("n_jobs", 1)))


def build_symbol_feature_frame(symbol: str, market_data: dict[str, dict[str, pd.DataFrame]], config: dict) -> pd.DataFrame:
    print(f"Building features for {symbol}...", flush=True)
    return build_symbol_dataset(symbol, market_data, config)


def build_feature_dataset(symbols: list[str], market_data: dict[str, dict[str, pd.DataFrame]], config: dict) -> pd.DataFrame:
    n_jobs = processing_jobs(config)
    if n_jobs == 1:
        frames = [build_symbol_feature_frame(symbol, market_data, config) for symbol in symbols]
    else:
        frames = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(build_symbol_feature_frame)(symbol, market_data, config)
            for symbol in symbols
        )
    dataset = pd.concat(frames, ignore_index=True).sort_values(["open_time", "symbol"])
    dataset["open_time"] = pd.to_datetime(dataset["open_time"], utc=True)
    if "close_time" in dataset.columns:
        dataset["close_time"] = pd.to_datetime(dataset["close_time"], utc=True)
    cols = feature_columns(dataset)
    dataset[cols] = dataset[cols].astype("float32")
    return dataset


def label_symbol_feature_frame(
    symbol: str,
    symbol_frame: pd.DataFrame,
    config: dict,
    confirmation_15m: pd.DataFrame | None,
) -> pd.DataFrame:
    print(f"Building labels for {symbol}...", flush=True)
    return add_long_labels(symbol_frame.sort_values("open_time"), config, confirmation_15m)


def apply_target_labels(
    feature_dataset: pd.DataFrame,
    config: dict,
    market_data: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    groups = list(feature_dataset.groupby("symbol", sort=False))
    n_jobs = processing_jobs(config)
    if n_jobs == 1:
        frames = [
            label_symbol_feature_frame(
                symbol,
                symbol_frame,
                config,
                market_data.get(symbol, {}).get("15m") if market_data is not None else None,
            )
            for symbol, symbol_frame in groups
        ]
    else:
        frames = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(label_symbol_feature_frame)(
                symbol,
                symbol_frame,
                config,
                market_data.get(symbol, {}).get("15m") if market_data is not None else None,
            )
            for symbol, symbol_frame in groups
        )
    dataset = pd.concat(frames, ignore_index=True).sort_values(["open_time", "symbol"])
    cols = feature_columns(dataset)
    dataset = dataset.dropna(subset=cols + ["target", "forward_pnl", "exit_time"])
    dataset[cols] = dataset[cols].astype("float32")
    return dataset


def build_dataset(symbols: list[str], market_data: dict[str, dict[str, pd.DataFrame]], config: dict) -> pd.DataFrame:
    return apply_target_labels(build_feature_dataset(symbols, market_data, config), config, market_data)


def load_symbols(config: dict) -> list[str]:
    symbols_path = Path(config["data"]["symbols_file"])
    if not symbols_path.exists():
        raise FileNotFoundError(
            f"Missing {symbols_path}. Run `python -m xgb_trader.pipeline --mode download` first."
        )
    return json.loads(symbols_path.read_text(encoding="utf-8"))


def train_and_report(
    dataset: pd.DataFrame,
    symbols: list[str],
    config: dict,
    label_optimization_result: dict | None = None,
) -> None:
    train, test, backtest = chronological_split(dataset, config)
    model, columns, test_probabilities = train_model(train, test, config)
    threshold_metrics = optimize_threshold(
        test,
        test_probabilities,
        config["backtest"]["fee_rate"],
        config["model"].get("min_recall_for_threshold", 0.05),
        config,
    )
    save_artifacts(model, columns, threshold_metrics, config)

    backtest_probabilities = model.predict_proba(backtest[columns])[:, 1]
    trades, report = run_backtest(
        backtest,
        backtest_probabilities,
        threshold_metrics["threshold"],
        config["backtest"]["fee_rate"],
        config["backtest"]["max_concurrent_positions"],
        config["backtest"]["fixed_margin_usdt"],
        config["backtest"]["leverage"],
        config["backtest"]["initial_balance_usdt"],
        config,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    trades.to_csv(reports_dir / "backtest_trades.csv", index=False)
    (reports_dir / "threshold_metrics.json").write_text(json.dumps(threshold_metrics, indent=2), encoding="utf-8")
    (reports_dir / "backtest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if label_optimization_result is not None:
        (reports_dir / "label_optimization.json").write_text(
            json.dumps(label_optimization_result, indent=2),
            encoding="utf-8",
        )
    joblib.dump({"symbols": symbols, "config": config}, Path(config["model"]["output_dir"]) / "metadata.joblib")

    print("Selected symbols:", ", ".join(symbols))
    print("Rows:", len(dataset), "Train:", len(train), "Test:", len(test), "Backtest:", len(backtest))
    if label_optimization_result is not None:
        print("Selected TP/SL:", label_optimization_result)
    print("Threshold metrics:", threshold_metrics)
    print("Backtest report:", report)


def load_processed_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_path}. Run a full/train_only pipeline once to create the labeled dataset."
        )
    dataset = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time", "exit_time"])
    dataset["open_time"] = pd.to_datetime(dataset["open_time"], utc=True)
    if "close_time" in dataset.columns:
        dataset["close_time"] = pd.to_datetime(dataset["close_time"], utc=True)
    if "exit_time" in dataset.columns:
        dataset["exit_time"] = pd.to_datetime(dataset["exit_time"], utc=True)
    return dataset


def backtest_only(dataset_path: Path, config: dict) -> None:
    model_dir = Path(config["model"]["output_dir"])
    model_path = model_dir / "xgb_model.joblib"
    columns_path = model_dir / "feature_columns.joblib"
    metadata_path = model_dir / "metadata.joblib"
    missing = [path for path in (model_path, columns_path, metadata_path) if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model artifact(s): {missing_list}. Train the model before --mode backtest_only.")

    dataset = load_processed_dataset(dataset_path)
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    metadata = joblib.load(metadata_path)
    symbols = metadata.get("symbols", [])
    trained_config = metadata.get("config", {})

    # Keep model/data feature assumptions from training, while allowing the current
    # config to override threshold and backtest controls for fast iteration.
    if trained_config.get("features"):
        config["features"] = trained_config["features"]
    if trained_config.get("label"):
        config["label"] = trained_config["label"]

    _, test, backtest = chronological_split(dataset, config)
    missing_columns = [col for col in columns if col not in dataset.columns]
    if missing_columns:
        raise KeyError(f"Processed dataset is missing model feature columns: {missing_columns[:10]}")

    test_probabilities = model.predict_proba(test[columns])[:, 1]
    threshold_metrics = optimize_threshold(
        test,
        test_probabilities,
        config["backtest"]["fee_rate"],
        config["model"].get("min_recall_for_threshold", 0.05),
        config,
    )
    joblib.dump(threshold_metrics, model_dir / "threshold.joblib")

    backtest_probabilities = model.predict_proba(backtest[columns])[:, 1]
    trades, report = run_backtest(
        backtest,
        backtest_probabilities,
        threshold_metrics["threshold"],
        config["backtest"]["fee_rate"],
        config["backtest"]["max_concurrent_positions"],
        config["backtest"]["fixed_margin_usdt"],
        config["backtest"]["leverage"],
        config["backtest"]["initial_balance_usdt"],
        config,
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    trades.to_csv(reports_dir / "backtest_trades.csv", index=False)
    (reports_dir / "threshold_metrics.json").write_text(json.dumps(threshold_metrics, indent=2), encoding="utf-8")
    (reports_dir / "backtest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Backtest-only mode")
    if symbols:
        print("Selected symbols:", ", ".join(symbols))
    print("Rows:", len(dataset), "Test:", len(test), "Backtest:", len(backtest))
    print("Threshold metrics:", threshold_metrics)
    print("Backtest report:", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["full", "download", "train_only", "backtest_only"],
        default="full",
        help=(
            "full: incremental cache update then train; download: update cache only; "
            "train_only: no Binance API calls; backtest_only: reuse existing model and processed dataset."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Deprecated alias for --mode train_only.",
    )
    parser.add_argument(
        "--use-processed-dataset",
        action="store_true",
        help="Train directly from data/processed/dataset.csv without rebuilding features from raw cache.",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    ensure_dirs(config)
    mode = "train_only" if args.skip_download else args.mode
    dataset_path = Path(config["data"]["dataset_dir"]) / "dataset.csv"

    if mode == "backtest_only":
        backtest_only(dataset_path, config)
        return

    if mode in {"full", "download"}:
        client = BinanceFuturesClient(
            config["data"]["base_url"],
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_API_SECRET"),
            config["data"]["min_request_interval_seconds"],
        )
        symbols, market_data = download_market_data(config, client)
        if mode == "download":
            print("Cache updated for:", ", ".join(["BTCUSDT", *symbols]))
            return
    else:
        symbols = load_symbols(config)
        market_data = None

    if args.use_processed_dataset:
        dataset = load_processed_dataset(dataset_path)
        label_optimization_result = None
        saved_label_path = Path("reports") / "label_optimization.json"
        if saved_label_path.exists():
            label_optimization_result = json.loads(saved_label_path.read_text(encoding="utf-8"))
            config.setdefault("label", {})["take_profit_pct"] = label_optimization_result["tp_pct"]
            config.setdefault("label", {})["stop_loss_pct"] = label_optimization_result["sl_pct"]
    else:
        if market_data is None:
            market_data = load_cached_market_data(symbols, config)
        feature_dataset = build_feature_dataset(symbols, market_data, config)
        label_optimization_result = None
        if config.get("label_optimization", {}).get("enabled", False):
            label_optimization_result = optimize_label_targets(feature_dataset, config)
            config.setdefault("label", {})["take_profit_pct"] = label_optimization_result["tp_pct"]
            config.setdefault("label", {})["stop_loss_pct"] = label_optimization_result["sl_pct"]
        dataset = apply_target_labels(feature_dataset, config, market_data)
        dataset.to_csv(dataset_path, index=False)

    train_and_report(dataset, symbols, config, label_optimization_result)


if __name__ == "__main__":
    main()
