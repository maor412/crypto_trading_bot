# XGBoost Crypto Futures Trader

Long-only Binance USDT-M Futures research and live-execution scaffold.

The pipeline:

1. Selects the top 35 USDT perpetual altcoins by Binance Futures 24h quote volume, excluding BTC.
2. Downloads and caches 2 years of klines for 4h, 1h, 30m, 15m, and 5m for those symbols plus BTC.
3. Builds leakage-safe multi-timeframe features on a 5m decision clock.
4. Adds BTC market-state features at the same timestamps.
5. Labels long setups as `1` when +2% TP is hit before -1% SL, using 5m high/low path scanning.
6. Holds out the final month completely for backtesting.
7. Trains XGBoost on pre-backtest data, optimizes the decision threshold on the chronological test fold, then backtests on the isolated final month.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## CLI modes

```powershell
python -m xgb_trader.pipeline --config config.yaml --mode full
```

`--mode full` updates the local kline cache incrementally, rebuilds features, trains, optimizes the threshold, and backtests.

Parallelism is capped in `config.yaml`:

- `data.download_workers: 2` downloads or updates two symbols at a time to limit Binance rate-limit risk.
- `processing.n_jobs: 4` builds features and labels across symbols using threads, avoiding large dataframe copies.

```powershell
python -m xgb_trader.pipeline --config config.yaml --mode download
```

`--mode download` only updates `data/raw`. If CSV files already exist, the downloader reads them first and requests only missing candles from Binance.

```powershell
python -m xgb_trader.pipeline --config config.yaml --mode train_only
```

`--mode train_only` makes no Binance API calls. It rebuilds features, labels, model, threshold, and backtest strictly from the local raw cache.

```powershell
python -m xgb_trader.pipeline --config config.yaml --mode train_only --use-processed-dataset
```

This is the fastest rerun path. It trains directly from `data/processed/dataset.csv` without querying Binance and without rebuilding indicators.
Because this path reuses an already labeled dataset, it does not rerun TP/SL label optimization.

```powershell
python -m xgb_trader.pipeline --config config.yaml --mode backtest_only
```

This skips label generation and model training entirely. It loads the existing model artifacts from `models/`, reloads `data/processed/dataset.csv`, recalculates the Test-set threshold using the current `config.yaml` threshold constraints, and reruns the final leveraged backtest.

The old `--skip-download` flag is still supported as an alias for `--mode train_only`.

## Modeling notes

- All predictions are on a 5-minute decision clock.
- All indicator columns are shifted by one candle before merging.
- BTC market regime features are added from the BTC 4h chart: 50 EMA distance, 200 EMA distance, and 50 EMA above/below 200 EMA.
- The final month is isolated before training and threshold selection.
- If `label_optimization.enabled` is true, TP/SL candidates are selected before final training using only the final Train region. The optimizer splits that Train region again into inner Train/Validation data, trains lightweight XGBoost models in parallel, and chooses the pair with the best fixed-margin leveraged Validation PnL.
- Final threshold optimization uses a sniper rule: maximize Test-set Profit Factor only when precision, minimum trade count, and Profit Factor constraints are satisfied. If no threshold qualifies, it falls back to a hardened high threshold.
- Long labels use 5m path scanning as the authoritative order-resolution path and record a 15m path check when cached 15m data is present.
- Backtesting uses fixed-margin leveraged portfolio simulation by default: up to 6 concurrent positions, $15 margin per position, 10x leverage, and fees charged on leveraged notional.

TP/SL optimization uses:

- `label_optimization.tp_candidates`
- `label_optimization.sl_candidates`
- only valid pairs where `TP > SL`
- `joblib.Parallel` with `n_jobs: -2`
- per-worker XGBoost thread limiting to avoid CPU thrashing

The selected pair is saved to `reports/label_optimization.json` and embedded in `models/metadata.joblib` for live trading bracket orders.

Backtest report fields include:

- `leveraged_pnl_usdt`
- `leveraged_pnl_pct`
- `ending_balance_usdt`
- `win_rate`
- `total_trades`
- `max_drawdown`
- `total_fees_usdt`

## Live trading

Live trading is intentionally not started by the training script. To run the continuous live trader:

```powershell
$env:LIVE_TRADING="true"
$env:I_UNDERSTAND_THIS_PLACES_REAL_ORDERS="true"
python -m xgb_trader.live --config config.yaml
```

The live runner places real Binance Futures orders only when both switches are set and valid API keys are present. It is long-only and uses market entries with reduce-only TP/SL exit orders.

## Important risk note

This is execution software, not a profitability guarantee. Crypto futures are high risk, and the live mode can lose money quickly. Run the offline pipeline and inspect reports before enabling live orders.
