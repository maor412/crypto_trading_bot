from __future__ import annotations

import numpy as np
import pandas as pd


def btc_regime_column(config: dict | None) -> str | None:
    if not config:
        return None
    regime = config.get("features", {}).get("btc_market_regime", {})
    if not regime.get("enabled", False):
        return None
    return f"btc_{regime.get('timeframe', '4h')}_ema_regime_bull"


def _release_closed_positions(
    open_positions: list[dict],
    now: pd.Timestamp,
    balance: float,
    equity_events: list[dict],
) -> tuple[list[dict], float]:
    remaining = []
    for position in open_positions:
        if position["exit_time"] <= now:
            balance += position["margin_usdt"] + position["net_pnl_usdt"]
            equity_events.append({"time": position["exit_time"], "balance": balance})
        else:
            remaining.append(position)
    return remaining, balance


def run_backtest(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    fee_rate: float,
    max_concurrent_positions: int = 1,
    fixed_margin_usdt: float = 15.0,
    leverage: float = 10.0,
    initial_balance_usdt: float = 1000.0,
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    data = frame.sort_values(["open_time", "symbol"]).copy()
    data["probability"] = probabilities
    regime_col = btc_regime_column(config)
    if regime_col is not None and regime_col not in data.columns:
        raise KeyError(f"BTC regime filter is enabled, but required feature column `{regime_col}` is missing.")

    trades = []
    regime_blocked_signals = 0
    blocked_until_by_symbol: dict[str, pd.Timestamp] = {}
    open_positions: list[dict] = []
    balance = float(initial_balance_usdt)
    equity_events = [{"time": data["open_time"].min() if not data.empty else pd.NaT, "balance": balance}]

    for row in data.itertuples(index=False):
        if pd.isna(row.exit_time):
            continue
        open_positions, balance = _release_closed_positions(open_positions, row.open_time, balance, equity_events)
        if len(open_positions) >= max_concurrent_positions:
            continue
        symbol = row.symbol
        blocked_until = blocked_until_by_symbol.get(symbol)
        if blocked_until is not None and row.open_time <= blocked_until:
            continue
        if row.probability < threshold:
            continue
        if regime_col is not None:
            regime_value = getattr(row, regime_col)
            if pd.isna(regime_value) or float(regime_value) < 0.5:
                regime_blocked_signals += 1
                continue

        if balance < fixed_margin_usdt:
            continue

        balance -= fixed_margin_usdt
        notional_usdt = fixed_margin_usdt * leverage
        gross_pnl_usdt = notional_usdt * float(row.forward_pnl)
        fees_usdt = notional_usdt * fee_rate * 2
        net_pnl_usdt = gross_pnl_usdt - fees_usdt
        leveraged_return_on_margin = net_pnl_usdt / fixed_margin_usdt
        equity_events.append({"time": row.open_time, "balance": balance})

        position = {
            "symbol": symbol,
            "entry_time": row.open_time,
            "exit_time": row.exit_time,
            "entry_price": row.close,
            "margin_usdt": fixed_margin_usdt,
            "notional_usdt": notional_usdt,
            "gross_pnl_usdt": gross_pnl_usdt,
            "fees_usdt": fees_usdt,
            "net_pnl_usdt": net_pnl_usdt,
            "leveraged_return_on_margin": leveraged_return_on_margin,
            "win": net_pnl_usdt > 0,
            "probability": row.probability,
        }
        trades.append(
            {
                **position,
                "balance_after_entry": balance,
            }
        )
        open_positions.append(position)
        blocked_until_by_symbol[symbol] = row.exit_time

    for position in sorted(open_positions, key=lambda item: item["exit_time"]):
        balance += position["margin_usdt"] + position["net_pnl_usdt"]
        equity_events.append({"time": position["exit_time"], "balance": balance})

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        return trade_frame, {
            "total_trades": 0,
            "trades": 0,
            "leveraged_pnl_usdt": 0.0,
            "leveraged_pnl_pct": 0.0,
            "ending_balance_usdt": float(initial_balance_usdt),
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "max_concurrent_positions": int(max_concurrent_positions),
            "fixed_margin_usdt": float(fixed_margin_usdt),
            "leverage": float(leverage),
            "btc_regime_filter_enabled": regime_col is not None,
            "btc_regime_column": regime_col,
            "regime_blocked_signals": int(regime_blocked_signals),
        }

    equity = pd.DataFrame(equity_events).sort_values("time")
    equity["peak"] = equity["balance"].cummax()
    drawdown = equity["balance"] / equity["peak"] - 1
    leveraged_pnl_usdt = balance - initial_balance_usdt
    report = {
        "total_trades": int(len(trade_frame)),
        "trades": int(len(trade_frame)),
        "leveraged_pnl_usdt": float(leveraged_pnl_usdt),
        "leveraged_pnl_pct": float(leveraged_pnl_usdt / initial_balance_usdt),
        "ending_balance_usdt": float(balance),
        "win_rate": float(trade_frame["win"].mean()),
        "max_drawdown": float(drawdown.min()),
        "max_concurrent_positions": int(max_concurrent_positions),
        "fixed_margin_usdt": float(fixed_margin_usdt),
        "leverage": float(leverage),
        "total_fees_usdt": float(trade_frame["fees_usdt"].sum()),
        "btc_regime_filter_enabled": regime_col is not None,
        "btc_regime_column": regime_col,
        "regime_blocked_signals": int(regime_blocked_signals),
    }
    return trade_frame, report
