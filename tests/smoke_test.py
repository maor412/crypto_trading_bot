from __future__ import annotations

import numpy as np
import pandas as pd

from xgb_trader.backtest import run_backtest
from xgb_trader.features import build_symbol_dataset
from xgb_trader.labels import add_long_labels
from xgb_trader.modeling import chronological_split, optimize_threshold, train_model
from xgb_trader.optimize_targets import optimize_label_targets


def make_klines(periods: int, freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    returns = rng.normal(0.0002, 0.004, periods)
    close = 100 * np.cumprod(1 + returns)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.004, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.004, periods))
    volume = rng.uniform(1000, 5000, periods)
    return pd.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": times + pd.to_timedelta(pd.Timedelta(freq)),
            "quote_volume": volume * close,
            "trade_count": rng.integers(100, 500, periods),
            "taker_buy_base_volume": volume * 0.5,
            "taker_buy_quote_volume": volume * close * 0.5,
        }
    )


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    indexed = frame.set_index("open_time")
    out = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna().reset_index()
    out["close_time"] = out["open_time"] + pd.to_timedelta(pd.Timedelta(rule))
    out["quote_volume"] = out["volume"] * out["close"]
    out["trade_count"] = 100
    out["taker_buy_base_volume"] = out["volume"] * 0.5
    out["taker_buy_quote_volume"] = out["quote_volume"] * 0.5
    return out


def main() -> None:
    config = {
        "data": {"timeframes": ["4h", "1h", "30m", "15m", "5m"], "base_timeframe": "5m"},
        "features": {
            "lag_periods": [1, 2, 3],
            "indicator_windows": {"rsi": 14, "atr": 14, "bollinger": 20, "volume_profile": 48},
        },
        "label": {"max_holding_bars_5m": 24, "take_profit_pct": 0.02, "stop_loss_pct": 0.01},
        "label_optimization": {
            "enabled": True,
            "tp_candidates": [0.015, 0.02],
            "sl_candidates": [0.01],
            "n_jobs": -2,
            "lightweight_estimators": 10,
            "validation_splits": 2,
        },
        "split": {"backtest_months": 1, "n_time_series_splits": 3},
        "model": {
            "random_state": 7,
            "params": {
                "n_estimators": 10,
                "max_depth": 2,
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
            },
        },
        "backtest": {
            "fee_rate": 0.0004,
            "initial_balance_usdt": 1000,
            "fixed_margin_usdt": 15,
            "leverage": 10,
            "max_concurrent_positions": 6,
        },
    }
    base_alt = make_klines(22000, "5min", 1)
    base_btc = make_klines(22000, "5min", 2)
    market_data = {"ETHUSDT": {}, "BTCUSDT": {}}
    for symbol, base in [("ETHUSDT", base_alt), ("BTCUSDT", base_btc)]:
        market_data[symbol]["5m"] = base
        market_data[symbol]["15m"] = resample_ohlcv(base, "15min")
        market_data[symbol]["30m"] = resample_ohlcv(base, "30min")
        market_data[symbol]["1h"] = resample_ohlcv(base, "1h")
        market_data[symbol]["4h"] = resample_ohlcv(base, "4h")

    feature_dataset = build_symbol_dataset("ETHUSDT", market_data, config)
    selected = optimize_label_targets(feature_dataset, config)
    config["label"]["take_profit_pct"] = selected["tp_pct"]
    config["label"]["stop_loss_pct"] = selected["sl_pct"]
    dataset = add_long_labels(feature_dataset, config, market_data["ETHUSDT"]["15m"]).dropna()
    train, test, backtest = chronological_split(dataset, config)
    model, columns, probs = train_model(train, test, config)
    threshold = optimize_threshold(test, probs, config["backtest"]["fee_rate"], config=config)
    bt_probs = model.predict_proba(backtest[columns])[:, 1]
    _, report = run_backtest(
        backtest,
        bt_probs,
        threshold["threshold"],
        config["backtest"]["fee_rate"],
        config["backtest"]["max_concurrent_positions"],
        config["backtest"]["fixed_margin_usdt"],
        config["backtest"]["leverage"],
        config["backtest"]["initial_balance_usdt"],
        config,
    )
    print({"rows": len(dataset), "selected": selected, "threshold": threshold, "backtest": report})


if __name__ == "__main__":
    main()
