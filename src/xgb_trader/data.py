from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .binance_client import BinanceFuturesClient


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


NUMERIC_KLINE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def normalize_kline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    df = frame.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    numeric_cols = [col for col in NUMERIC_KLINE_COLUMNS if col in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return (
        df.dropna(subset=["open_time", "close_time"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def select_top_altcoin_symbols(client: BinanceFuturesClient, top_n: int, symbols_file: str | Path) -> list[str]:
    symbols_path = Path(symbols_file)
    tickers = client.get("/fapi/v1/ticker/24hr")
    exchange_info = client.get("/fapi/v1/exchangeInfo")
    valid = {
        item["symbol"]
        for item in exchange_info["symbols"]
        if item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
        and item.get("status") == "TRADING"
    }
    excluded_prefixes = ("BTC",)
    excluded_tokens = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
    ranked = sorted(
        (
            row
            for row in tickers
            if row["symbol"] in valid
            and row["symbol"].endswith("USDT")
            and not row["symbol"].startswith(excluded_prefixes)
            and not row["symbol"].endswith(excluded_tokens)
        ),
        key=lambda row: float(row.get("quoteVolume", 0.0)),
        reverse=True,
    )
    symbols = [row["symbol"] for row in ranked[:top_n]]
    symbols_path.parent.mkdir(parents=True, exist_ok=True)
    symbols_path.write_text(json.dumps(symbols, indent=2), encoding="utf-8")
    return symbols


def fetch_klines(
    client: BinanceFuturesClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    rows: list[list[object]] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = client.get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(client.min_interval_seconds)

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        return frame
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return normalize_kline_frame(frame.drop(columns=["ignore"]))


def cached_klines(
    client: BinanceFuturesClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    cache_dir: str | Path,
) -> pd.DataFrame:
    path = Path(cache_dir) / symbol / f"{interval}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = normalize_kline_frame(pd.read_csv(path))
        if not frame.empty:
            have_start = int(frame["open_time"].min().timestamp() * 1000)
            have_end = int(frame["open_time"].max().timestamp() * 1000)
            pieces = [frame]
            if have_start > start_ms:
                pieces.append(fetch_klines(client, symbol, interval, start_ms, have_start - 1))
            if have_end < end_ms - 1:
                pieces.append(fetch_klines(client, symbol, interval, have_end + 1, end_ms))
            frame = pd.concat(pieces, ignore_index=True)
            frame = normalize_kline_frame(frame)
            frame.to_csv(path, index=False)
            return frame

    frame = fetch_klines(client, symbol, interval, start_ms, end_ms)
    frame.to_csv(path, index=False)
    return frame


def download_market_data(config: dict, client: BinanceFuturesClient) -> tuple[list[str], dict[str, dict[str, pd.DataFrame]]]:
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=config["data"]["lookback_days"])
    start_ms, end_ms = utc_ms(start), utc_ms(end)

    symbols_path = Path(config["data"]["symbols_file"])
    if symbols_path.exists() and not config["data"].get("refresh_top_symbols", False):
        symbols = json.loads(symbols_path.read_text(encoding="utf-8"))
    else:
        symbols = select_top_altcoin_symbols(client, config["data"]["top_n_altcoins"], symbols_path)
    all_symbols = ["BTCUSDT", *symbols]
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in all_symbols:
        data[symbol] = {}
        for timeframe in config["data"]["timeframes"]:
            data[symbol][timeframe] = cached_klines(
                client,
                symbol,
                timeframe,
                start_ms,
                end_ms,
                config["data"]["cache_dir"],
            )
    return symbols, data


def load_cached_market_data(symbols: Iterable[str], config: dict) -> dict[str, dict[str, pd.DataFrame]]:
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in ["BTCUSDT", *symbols]:
        data[symbol] = {}
        for timeframe in config["data"]["timeframes"]:
            path = Path(config["data"]["cache_dir"]) / symbol / f"{timeframe}.csv"
            data[symbol][timeframe] = normalize_kline_frame(pd.read_csv(path))
    return data
