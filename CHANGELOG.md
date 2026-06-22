# CHANGELOG

## v0.10.1 (2026-06-22)

`Wallet` class audited after the panzer credential migration; two pre-existing bugs fixed and a
self-contained test notebook added.

### Fixed

- **`Wallet` snapshot methods were unreachable.** `spot_snapshot`/`margin_snapshot` collided with
  instance attributes set to `None` in `__init__`, shadowing the methods (`TypeError: 'NoneType' object
  is not callable`). Renamed to `update_spot_snapshot`/`update_margin_snapshot`; the `spot_snapshot`/
  `margin_snapshot` attributes now hold the snapshot DataFrames. `update_margin_snapshot` also wrote to
  `self.spot` (wrong side) and both returned balances instead of the snapshot — corrected.
- **`spot_wallet_performance`/`margin_wallet_performance` clobbered balances.** They overwrote
  `self.spot`/`self.margin` with snapshot rows and raised `KeyError('totalAssetOfBtc')` on a cache hit.
  They now use the snapshot attribute as cache and leave the balances untouched.

### Added

- `notebooks/18_wallet_tests.ipynb`: self-contained `Wallet` tests that run without API keys (they mock
  `panzer.BinanceClient.signed_request`), covering the shadowing regression and the no-clobber invariant.

## v0.10.0 (2026-06-18)

Credential management fully delegated to `panzer`, Telegram support removed, new `Trades` value
object, indicator/plotting fixes, and a notebook overhaul.

### Removed

- **Telegram support removed entirely** (`binpan/core/messages.py`): `telegram_bot_send_text`,
  `telegram_parse_*`, `send_balances`, `sort_mixed_dict`, `tab_str`, the `MissingTelegramApiData`
  exception and `get_telegram_secrets()`.
- `binpan/core/crypto.py` removed — all secret management is delegated to `panzer`.
- Dead code removed: `time_helper.ceil_division`, `detect_tick_interval`, `time_interval`, the unused
  `Trades(Timeframe)` class, and 9 unused exception classes (`MissingBinanceApiData`,
  `BinanceAPIException`, `BinanceRequestException`, `BinanceOrderException` + subclasses,
  `NotImplementedException`).
- `pycryptodome` and `py-cpuinfo` dropped from requirements (only used by the removed `crypto.py`;
  `panzer` provides `pycryptodome` transitively).

### Added

- `binpan/core/secrets.py`: thin wrapper over `panzer`'s `CredentialManager`
  (`get_secret`/`set_secret`/`get_json_secret`/`set_json_secret`).
- `Trades` value object (`binpan/core/trades.py`): wraps a trades DataFrame plus metadata (trade type,
  origin, columns) and proxies attribute/item access to the DataFrame. `Symbol.agg_trades` /
  `atomic_trades` are now `Trades` instances (backward compatible via the proxy).
- New notebook `17_credentials_and_panzer` documenting credential setup. `nbstripout` filter added so
  notebook outputs are stripped from git.

### Changed

- All credentials (Binance API keys, PostgreSQL/binbase passwords, Redis configs) are now managed by
  `panzer` (`~/.panzer_creds`); BinPan no longer implements its own encryption. `create_connection`
  takes a plain-text password.
- Trade-fetching consumers deduplicated through the `Trades` selection helpers.
- Notebooks reordered and cleaned (01, 02, 04); trade-precision examples use a dedicated intraday symbol.

### Fixed

- `ker()` used the removed `pd.rolling_sum` → `AttributeError` on modern pandas.
- `sma_numba` returned partial averages during the warm-up window instead of `NaN`.
- `get_agg_trades`/`get_atomic_trades` with `hours=`/`minutes=` raised `startTime > endTime` when the
  Symbol's `end_time` was in the past; the window is now anchored to the end of the data.
- `Symbol(from_csv=...)` did not initialize the connection state, so `get_*_trades(from_csv=...)` raised
  `AttributeError: cursor_agg_trades`.
- Unnamed `pd.Series` passed to `candles_ta` no longer appear as `"trace N"` in the legend.
- `resample()` to a smaller interval now raises a clear message instead of a cryptic index comparison.
- Plot action markers: validation relaxed to require a label per present action (extra labels ignored)
  instead of an exact length match.

## v0.9.9 (2026-06-16)

Plot legend/series alignment fixes on the candle charts.

### Fixed

- `Symbol.plot` legend/colors mismatch when support/resistance lines were drawn together with
  indicators: overlapped series (supports, resistances and any `overlapped_indicators`) are
  plotted before the row indicators, so their names and colors are now prepended to stay
  aligned. Previously the RSI could be painted with a support color, a support line could take
  the RSI color, and the legend did not match the lines. This also prevented the cloud-fill
  (`plot_splitted_serie_couple`) of MACD/Ichimoku/Bollinger from being triggered on the wrong
  indicator when support/resistance lines were present.
- `plotting.charts.candles_ta` created an empty trailing subplot row whenever a row-1 overlay
  indicator (EMA, Bollinger, Ichimoku, VWAP, Supertrend) or support/resistance lines were
  present: row 1 is the candles overlay, not a subplot of its own, so only rows `> 1` are now
  counted as extra subplots.

## v0.9.8 (2026-06-14)

Operation markers (▲ buy / ▼ sell) on the candle charts.

### Features

- `plotting.charts.set_price_markers(markers, klines_index)`: builds triangle marker traces
  from an explicit list of priced points (`{'time', 'price', 'side', 'label'?}`). `time` is a
  positional candle index (int) or a timestamp snapped to the nearest candle; `price` is the
  exact y level; `side` 'buy' → green ▲ (label below), otherwise red ▼ (label above).
- `Symbol.plot(..., priced_markers=...)` and `plotting.charts.candles_tagged`/`candles_ta`:
  overlay operation markers on exact points over the candles. Compatible with
  support/resistance lines and action columns (they overlay together).
- `Symbol.plot_volume_profile(..., priced_markers=...)` and the underlying
  `plotting.charts.plot_volume_profile(...)`: same markers on the candle panel of the VPVR.

## v0.9.7 (2026-06-14)

Volume Profile (VPVR): analytics + composite chart.

### Features

- `analysis.indicators.value_area_from_profile(profile, value_area_pct=0.70)`: computes
  POC, Value Area (VAH/VAL) and HVN/LVN nodes from a market-profile DataFrame.
- `Symbol.volume_profile(bins=50, value_area_pct=0.70, from_agg_trades=..., ...)`: returns
  the volume profile numbers (POC/VAH/VAL/HVN/LVN + per-bin volumes). Reuses
  `get_market_profile` for klines/trades sourcing and windowing.
- `Symbol.plot_volume_profile(...)`: composite VPVR chart — candlesticks + horizontal
  volume histogram sharing the price axis, POC line, shaded Value Area, LVN marks.
  Headless export (`show=False`, `image_path=`, `width=`) and optional `horizontal_lines`.
- `plotting.charts.plot_volume_profile(...)`: the underlying figure builder.

## v0.9.6 (2026-06-14)

Headless export for the trades scatter plot.

### Features

- `plot_trades()` (`plotting/charts.py`) no longer forces `fig.show()` nor writes
  to a fixed `last_plot.png`. New params: `show` (default `True`; set `False` for
  servers/headless), `image_path` (PNG output path), `width`, `size_column`
  (size bubbles by any column, e.g. a computed `Quote`), and `horizontal_lines`
  (draw price level lines). Returns the absolute path of the exported image.
- Threaded through `Symbol._plot_trades_size()`, `Symbol.plot_agg_trades_size()`
  and `Symbol.plot_atomic_trades_size()`.

## v0.9.5 (2026-06-13)

Headless export for the candles plot (programmatic / server use).

### Features

- `Symbol.plot()` and the candles plotting chain (`candles_ta`, `candles_tagged`,
  `_finalize_and_export_figure` in `plotting/charts.py`) accept `show: bool = True`
  and `image_path: str = None`. With `show=False` no interactive figure is opened;
  `image_path` sets the PNG output path (defaults to `last_plot.png` in the cwd).
  Enables generating chart images on a headless host without a browser/display.

## v0.9.4 (2026-03-14)

Documentation.

### Changed

- `binbase` notebooks refreshed with real execution outputs.

## v0.9.2 (2026-03-07)

Full code review (P0–P8) and CI hardening.

### Changed

- Security, exception handling, pandas deprecations, overly long functions,
  duplication and type hints reviewed across the codebase.
- Single source of truth for the version; tests and CI added.

## v0.9.1 (2026-03-06)

### Changed

- Version bump for PyPI publication.

## v0.9.0 (2026-03-05)

First public 0.9 release.
