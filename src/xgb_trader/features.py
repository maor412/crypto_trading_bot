from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_utc_open_time(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    return df.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(frame: pd.DataFrame, config: dict, prefix: str) -> pd.DataFrame:
    windows = config["features"]["indicator_windows"]
    df = ensure_utc_open_time(frame)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / windows["atr"], adjust=False).mean()

    bb_mid = close.rolling(windows["bollinger"]).mean()
    bb_std = close.rolling(windows["bollinger"]).std()
    vp_window = windows["volume_profile"]
    rolling_vwap = (close * volume).rolling(vp_window).sum() / volume.rolling(vp_window).sum()

    out = pd.DataFrame(
        {
            "open_time": df["open_time"],
            f"{prefix}_ret_1": close.pct_change(),
            f"{prefix}_range_pct": (high - low) / close,
            f"{prefix}_volume_chg": volume.pct_change(),
            f"{prefix}_rsi": rsi(close, windows["rsi"]),
            f"{prefix}_macd": macd / close,
            f"{prefix}_macd_signal": signal / close,
            f"{prefix}_macd_hist": (macd - signal) / close,
            f"{prefix}_atr_pct": atr / close,
            f"{prefix}_bb_width": (4 * bb_std) / close,
            f"{prefix}_bb_pos": (close - bb_mid) / (2 * bb_std),
            f"{prefix}_vp_vwap_dist": (close - rolling_vwap) / close,
            f"{prefix}_volume_z": (volume - volume.rolling(vp_window).mean()) / volume.rolling(vp_window).std(),
        }
    )
    regime = config["features"].get("btc_market_regime", {})
    if (
        regime.get("enabled", False)
        and prefix == f"btc_{regime.get('timeframe', '4h')}"
    ):
        ema_fast = close.ewm(span=int(regime.get("ema_fast", 50)), adjust=False).mean()
        ema_slow = close.ewm(span=int(regime.get("ema_slow", 200)), adjust=False).mean()
        out[f"{prefix}_ema_{regime.get('ema_fast', 50)}_dist"] = (close - ema_fast) / close
        out[f"{prefix}_ema_{regime.get('ema_slow', 200)}_dist"] = (close - ema_slow) / close
        out[f"{prefix}_ema_regime_bull"] = (ema_fast > ema_slow).astype(float)

    for lag in config["features"]["lag_periods"]:
        out[f"{prefix}_ret_lag_{lag}"] = close.pct_change(lag)

    feature_cols = [col for col in out.columns if col != "open_time"]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).shift(1)
    return out


def merge_timeframes(symbol_frames: dict[str, pd.DataFrame], config: dict, symbol_prefix: str) -> pd.DataFrame:
    base_tf = config["data"]["base_timeframe"]
    base = ensure_utc_open_time(symbol_frames[base_tf])[
        ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    ]
    merged = base.copy()
    for timeframe in config["data"]["timeframes"]:
        features = add_indicators(symbol_frames[timeframe], config, f"{symbol_prefix}_{timeframe}")
        merged = pd.merge_asof(
            merged.sort_values("open_time"),
            features.sort_values("open_time"),
            on="open_time",
            direction="backward",
            allow_exact_matches=True,
        )
    return merged


def build_symbol_dataset(symbol: str, market_data: dict[str, dict[str, pd.DataFrame]], config: dict) -> pd.DataFrame:
    alt = merge_timeframes(market_data[symbol], config, "alt")
    btc = merge_timeframes(market_data["BTCUSDT"], config, "btc")
    btc_features = btc.drop(columns=["open", "high", "low", "close", "volume", "close_time"], errors="ignore")
    merged = pd.merge_asof(
        alt.sort_values("open_time"),
        btc_features.sort_values("open_time"),
        on="open_time",
        direction="backward",
        allow_exact_matches=True,
    ).copy()
    merged.insert(0, "symbol", symbol)
    return merged
