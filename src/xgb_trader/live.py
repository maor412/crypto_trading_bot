from __future__ import annotations

import argparse
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv

try:
    import telebot
except ImportError:  # Telegram is optional unless TELEGRAM_BOT_TOKEN is configured.
    telebot = None

from .binance_client import BinanceFuturesClient
from .config import load_config
from .data import fetch_klines
from .features import build_symbol_dataset


class LiveState:
    def __init__(self, client: BinanceFuturesClient, config: dict, max_positions: int) -> None:
        self.client = client
        self.config = config
        self.max_positions = max_positions
        self.paused = threading.Event()
        self.client_lock = threading.RLock()
        self.tracked_positions: dict[str, dict[str, Any]] = {}


class TelegramBridge:
    def __init__(self, state: LiveState) -> None:
        self.state = state
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        allowed = os.getenv("TELEGRAM_ALLOWED_USER_ID", "").strip()
        self.allowed_user_id = int(allowed) if allowed.isdigit() else None
        self.bot = telebot.TeleBot(self.token, parse_mode=None) if self.token and telebot else None

    @property
    def enabled(self) -> bool:
        return self.bot is not None and self.allowed_user_id is not None

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            self.bot.send_message(self.allowed_user_id, text)
        except Exception as exc:
            print(f"Telegram send failed: {exc}")

    def _authorized(self, message: Any) -> bool:
        try:
            return int(message.chat.id) == self.allowed_user_id
        except Exception:
            return False

    def _reply(self, message: Any, text: str) -> None:
        if not self._authorized(message):
            return
        try:
            self.bot.reply_to(message, text)
        except Exception as exc:
            print(f"Telegram reply failed: {exc}")

    def start_polling(self) -> None:
        if not self.enabled:
            if self.token and telebot is None:
                print("Telegram token configured, but pyTelegramBotAPI is not installed.")
            else:
                print("Telegram disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_ALLOWED_USER_ID missing.")
            return

        @self.bot.message_handler(commands=["status"])
        def status(message: Any) -> None:
            if not self._authorized(message):
                return
            try:
                with self.state.client_lock:
                    balance = futures_available_balance(self.state.client)
                    positions = active_positions(self.state.client)
                lines = [
                    f"Status: {'PAUSED' if self.state.paused.is_set() else 'RUNNING'}",
                    f"Free Balance: {balance:.2f} USDT",
                    f"Open: {len(positions)}/{self.state.max_positions}",
                ]
                if positions:
                    lines.append("Positions:")
                    for symbol, pos in sorted(positions.items()):
                        lines.append(f"{symbol}: qty={pos['amount']:.8g}, uPnL={pos['unrealized_pnl']:.2f} USDT")
                self._reply(message, "\n".join(lines))
            except Exception as exc:
                self._reply(message, f"Status error: {exc}")

        @self.bot.message_handler(commands=["pause"])
        def pause(message: Any) -> None:
            if not self._authorized(message):
                return
            self.state.paused.set()
            self._reply(message, "⏸️ Trading paused. Existing positions will be managed.")

        @self.bot.message_handler(commands=["resume"])
        def resume(message: Any) -> None:
            if not self._authorized(message):
                return
            self.state.paused.clear()
            self._reply(message, "▶️ Trading resumed.")

        @self.bot.message_handler(commands=["closeall"])
        def closeall(message: Any) -> None:
            if not self._authorized(message):
                return
            try:
                with self.state.client_lock:
                    positions = active_positions(self.state.client)
                    for symbol, position in positions.items():
                        close_position_market(self.state.client, symbol, position["amount"])
                    self.state.tracked_positions.clear()
                self._reply(message, "🛑 EMERGENCY: All positions closed.")
            except Exception as exc:
                self._reply(message, f"Close-all error: {exc}")

        def poll() -> None:
            while True:
                try:
                    self.bot.infinity_polling(timeout=20, long_polling_timeout=20, skip_pending=True)
                except Exception as exc:
                    print(f"Telegram polling failed: {exc}")
                    time.sleep(10)

        threading.Thread(target=poll, name="telegram-polling", daemon=True).start()


def btc_regime_column(config: dict) -> str | None:
    regime = config.get("features", {}).get("btc_market_regime", {})
    if not regime.get("enabled", False):
        return None
    return f"btc_{regime.get('timeframe', '4h')}_ema_regime_bull"


def latest_btc_regime_is_bullish(market_data: dict[str, dict[str, pd.DataFrame]], symbols: list[str], config: dict) -> bool:
    regime_col = btc_regime_column(config)
    if regime_col is None:
        return True
    for symbol in symbols:
        frame = build_symbol_dataset(symbol, market_data, config)
        if frame.empty or regime_col not in frame.columns:
            continue
        value = frame.iloc[-1][regime_col]
        if pd.isna(value):
            continue
        return float(value) >= 0.5
    raise RuntimeError(f"BTC regime filter is enabled, but `{regime_col}` could not be computed from live market data.")


def live_enabled() -> bool:
    return (
        os.getenv("LIVE_TRADING", "").lower() == "true"
        and os.getenv("I_UNDERSTAND_THIS_PLACES_REAL_ORDERS", "").lower() == "true"
    )


def round_quantity(quantity: float, step_size: float) -> float:
    precision = max(0, int(round(-math.log10(step_size))))
    return math.floor(quantity / step_size) * step_size if precision == 0 else round(math.floor(quantity / step_size) * step_size, precision)


def symbol_filters(client: BinanceFuturesClient, symbol: str) -> tuple[float, float]:
    info = client.get("/fapi/v1/exchangeInfo")
    item = next(row for row in info["symbols"] if row["symbol"] == symbol)
    lot = next(f for f in item["filters"] if f["filterType"] == "LOT_SIZE")
    price = next(f for f in item["filters"] if f["filterType"] == "PRICE_FILTER")
    return float(lot["stepSize"]), float(price["tickSize"])


def futures_available_balance(client: BinanceFuturesClient) -> float:
    account = client.get("/fapi/v2/account", signed=True)
    for asset in account.get("assets", []):
        if asset.get("asset") == "USDT":
            return float(asset.get("availableBalance", 0.0))
    return float(account.get("availableBalance", 0.0))


def active_positions(client: BinanceFuturesClient) -> dict[str, dict[str, float]]:
    positions = client.get("/fapi/v2/positionRisk", signed=True)
    active: dict[str, dict[str, float]] = {}
    for row in positions:
        amount = float(row.get("positionAmt", 0.0))
        if abs(amount) > 0:
            active[row["symbol"]] = {
                "amount": amount,
                "entry_price": float(row.get("entryPrice", 0.0)),
                "mark_price": float(row.get("markPrice", 0.0)),
                "unrealized_pnl": float(row.get("unRealizedProfit", 0.0)),
                "leverage": float(row.get("leverage", 0.0)),
                "update_time": float(row.get("updateTime", 0.0)),
            }
    return active


def configure_symbol_leverage(client: BinanceFuturesClient, symbol: str, leverage: int) -> None:
    client.post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})


def close_position_market(client: BinanceFuturesClient, symbol: str, amount: float) -> None:
    if amount == 0:
        return
    step_size, _ = symbol_filters(client, symbol)
    qty = round_quantity(abs(amount), step_size)
    if qty <= 0:
        raise RuntimeError(f"Computed non-positive close quantity for {symbol}.")
    side = "SELL" if amount > 0 else "BUY"
    client.post(
        "/fapi/v1/order",
        {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty, "reduceOnly": "true"},
    )


def realized_income_since(client: BinanceFuturesClient, symbol: str, start_time_ms: int) -> tuple[float, float]:
    rows = client.get("/fapi/v1/income", {"symbol": symbol, "startTime": start_time_ms, "limit": 1000}, signed=True)
    pnl = 0.0
    fees = 0.0
    for row in rows:
        income = float(row.get("income", 0.0))
        if row.get("incomeType") == "REALIZED_PNL":
            pnl += income
        elif row.get("incomeType") == "COMMISSION":
            fees += abs(income)
    return pnl, fees


def sync_position_tracking(
    state: LiveState,
    positions: dict[str, dict[str, float]],
    notifier: TelegramBridge,
) -> None:
    now_ms = int(time.time() * 1000)
    for symbol, position in positions.items():
        state.tracked_positions.setdefault(
            symbol,
            {
                "entry_time_ms": int(position.get("update_time") or now_ms),
                "entry_price": position.get("entry_price", 0.0),
            },
        )

    closed_symbols = [symbol for symbol in state.tracked_positions if symbol not in positions]
    for symbol in closed_symbols:
        tracked = state.tracked_positions.pop(symbol)
        try:
            with state.client_lock:
                pnl, fees = realized_income_since(state.client, symbol, int(tracked["entry_time_ms"]))
            status = "WIN" if pnl > 0 else "LOSS"
            notifier.send(f"Exit Alert\nSymbol: {symbol}\nStatus: {status}\nNet PnL: {pnl:.4f} USDT\nTotal Fees: {fees:.4f} USDT")
        except Exception as exc:
            print(f"Exit notification failed for {symbol}: {exc}")


def latest_market_data(client: BinanceFuturesClient, symbols: list[str], config: dict) -> dict[str, dict[str, pd.DataFrame]]:
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(pd.Timedelta(days=45).total_seconds() * 1000)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in ["BTCUSDT", *symbols]:
        data[symbol] = {}
        for timeframe in config["data"]["timeframes"]:
            data[symbol][timeframe] = fetch_klines(client, symbol, timeframe, start_ms, end_ms)
    return data


def place_long_with_brackets(
    client: BinanceFuturesClient,
    symbol: str,
    notional_usdt: float,
    leverage: int,
    tp_pct: float,
    sl_pct: float,
) -> dict[str, float]:
    configure_symbol_leverage(client, symbol, leverage)
    price = float(client.get("/fapi/v1/ticker/price", {"symbol": symbol})["price"])
    step_size, tick_size = symbol_filters(client, symbol)
    qty = round_quantity(notional_usdt / price, step_size)
    if qty <= 0:
        raise RuntimeError(f"Computed non-positive quantity for {symbol}.")

    order = client.post("/fapi/v1/order", {"symbol": symbol, "side": "BUY", "type": "MARKET", "quantity": qty})
    tp = round(price * (1 + tp_pct) / tick_size) * tick_size
    sl = round(price * (1 - sl_pct) / tick_size) * tick_size
    client.post(
        "/fapi/v1/order",
        {"symbol": symbol, "side": "SELL", "type": "TAKE_PROFIT_MARKET", "stopPrice": tp, "closePosition": "true"},
    )
    client.post(
        "/fapi/v1/order",
        {"symbol": symbol, "side": "SELL", "type": "STOP_MARKET", "stopPrice": sl, "closePosition": "true"},
    )
    entry_price = float(order.get("avgPrice") or price)
    return {"entry_price": entry_price, "quantity": qty, "notional_usdt": notional_usdt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    load_dotenv()
    if not live_enabled():
        raise SystemExit("Live trading is disabled. Set LIVE_TRADING=true and I_UNDERSTAND_THIS_PLACES_REAL_ORDERS=true.")

    config = load_config(args.config)
    model = joblib.load(Path(config["model"]["output_dir"]) / "xgb_model.joblib")
    columns = joblib.load(Path(config["model"]["output_dir"]) / "feature_columns.joblib")
    threshold = joblib.load(Path(config["model"]["output_dir"]) / "threshold.joblib")["threshold"]
    metadata = joblib.load(Path(config["model"]["output_dir"]) / "metadata.joblib")
    symbols = metadata["symbols"]
    trained_config = metadata.get("config", {})
    if trained_config.get("label"):
        config["label"] = trained_config["label"]
    if trained_config.get("features"):
        config["features"] = trained_config["features"]

    client = BinanceFuturesClient(
        config["data"]["base_url"],
        os.getenv("BINANCE_API_KEY"),
        os.getenv("BINANCE_API_SECRET"),
        config["data"]["min_request_interval_seconds"],
    )
    live_leverage = int(config.get("live", {}).get("leverage", config["backtest"].get("leverage", 1)))
    fixed_margin = float(config.get("live", {}).get("fixed_margin_usdt", config["backtest"].get("fixed_margin_usdt", 15)))
    notional = fixed_margin * live_leverage
    max_positions = int(config.get("live", {}).get("max_positions", 1))
    state = LiveState(client, config, max_positions)
    notifier = TelegramBridge(state)
    notifier.start_polling()

    try:
        with state.client_lock:
            balance = futures_available_balance(client)
            initial_positions = active_positions(client)
        sync_position_tracking(state, initial_positions, notifier)
        startup_message = (
            "✅ Bot initialized and ready.\n"
            f"Free USDT Balance: {balance:.2f}\n"
            f"Fixed Margin: {fixed_margin:.2f} USDT\n"
            f"Leverage: {live_leverage}x\n"
            f"Max Positions: {max_positions}"
        )
        print(startup_message)
        notifier.send(startup_message)
    except Exception as exc:
        notifier.send(f"Error Alert\nStartup account check failed: {exc}")
        raise

    while True:
        try:
            with state.client_lock:
                open_positions = active_positions(client)
            sync_position_tracking(state, open_positions, notifier)

            if state.paused.is_set():
                print("Trading paused. Existing positions are still monitored.")
                time.sleep(config["live"]["poll_seconds"])
                continue

            if len(open_positions) >= max_positions:
                print(f"Max live positions reached: {len(open_positions)}/{max_positions}. Open symbols: {sorted(open_positions)}")
                time.sleep(config["live"]["poll_seconds"])
                continue

            market_data = latest_market_data(client, symbols, config)
            if not latest_btc_regime_is_bullish(market_data, symbols, config):
                print("Blocked all live long signals: BTC regime is bearish.")
                time.sleep(config["live"]["poll_seconds"])
                continue

            candidates = []
            for symbol in symbols:
                if symbol in open_positions:
                    continue
                frame = build_symbol_dataset(symbol, market_data, config).dropna(subset=columns)
                if frame.empty:
                    continue
                row = frame.iloc[[-1]]
                probability = float(model.predict_proba(row[columns])[:, 1][0])
                candidates.append((probability, symbol))

            candidates.sort(reverse=True)
            if candidates and candidates[0][0] >= threshold:
                probability, symbol = candidates[0]
                with state.client_lock:
                    order_info = place_long_with_brackets(
                        client,
                        symbol,
                        notional,
                        live_leverage,
                        config["label"]["take_profit_pct"],
                        config["label"]["stop_loss_pct"],
                    )
                state.tracked_positions[symbol] = {
                    "entry_time_ms": int(time.time() * 1000),
                    "entry_price": order_info["entry_price"],
                }
                message = (
                    f"Entry Alert\n"
                    f"Symbol: {symbol}\n"
                    f"Side: Long\n"
                    f"Model Probability: {probability * 100:.2f}%\n"
                    f"Entry Price: {order_info['entry_price']:.8g}"
                )
                notifier.send(message)
                print(
                    f"Placed live long on {symbol} probability={probability:.4f} "
                    f"margin={fixed_margin:.2f}USDT leverage={live_leverage}x notional={notional:.2f}USDT"
                )
            else:
                print("No signal", candidates[:3])
            time.sleep(config["live"]["poll_seconds"])
        except Exception as exc:
            error_message = f"Error Alert\nLive loop error: {exc}"
            print(error_message)
            notifier.send(error_message)
            time.sleep(config["live"]["poll_seconds"])


if __name__ == "__main__":
    main()
