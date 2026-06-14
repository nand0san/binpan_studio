# CHANGELOG

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
