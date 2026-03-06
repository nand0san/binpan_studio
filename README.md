Welcome to BinPan's documentation!
==================================

[![PyPI version](https://img.shields.io/pypi/v/binpan.svg)](https://pypi.org/project/binpan/)
[![Python](https://img.shields.io/pypi/pyversions/binpan.svg)](https://pypi.org/project/binpan/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BinPan is a Python wrapper for the Binance API oriented towards market data analysis: candlestick (kline) data,
trades, technical indicators, plotting, and strategy backtesting.

It is intended to be useful in Jupyter Notebooks, the Python console, or any data pipeline.

**Requires Python >= 3.12.** Published on [PyPI](https://pypi.org/project/binpan/) under the MIT license.

Features
--------

- Fetch candlestick (OHLCV) data with timezone and indexing options.
- Fetch atomic and aggregated trades.
- Native technical indicators (EMA, SMA, RSI, MACD, Bollinger Bands, Supertrend, Ichimoku, VWAP, Stochastic, fractals, and more).
- Interactive candlestick charts with Plotly: indicators, buy/sell markers, trade bubble charts, order book, market profile.
- Support and resistance detection via K-Means clustering (static, rolling, discrete interval).
- Strategy tagging and backtesting engine with stop loss, target, entry filters, and ROI/profit metrics.
- Exchange metadata: order types, filters, fees, coin networks, 24h volume ranking.
- CSV export/import for klines and trades.
- PostgreSQL/TimescaleDB integration for historical data storage.
- Heikin-Ashi candles, reversal charts from tick data, kline resampling.

An example of a candlestick chart with indicators:

![](https://raw.githubusercontent.com/nand0san/binpan_studio/main/docs/images/candles.png)

> BinPan contains no Binance **order method, withdraw method** or any dangerous command.
>
> If you decide to add API keys for account methods, BinPan will encrypt them in a file and in memory,
> but it is recommended not to enable trading capability in your Binance API key configuration.
>
> Be careful out there!


Installation
------------

```bash
pip install binpan
```

Any API key or secret will be prompted when needed and encrypted in a file.

Quick Start
-----------

```python
import binpan

btcusdt = binpan.Symbol(symbol='btcusdt',
                        tick_interval='15m',
                        time_zone='Europe/Madrid',
                        limit=200)

btcusdt.ema(21)
btcusdt.supertrend()
btcusdt.plot()
```

![](https://raw.githubusercontent.com/nand0san/binpan_studio/main/docs/images/candles_ta.png)

Example Notebooks
-----------------

The `notebooks/` folder contains example Jupyter Notebooks organized by topic:

| # | Notebook | Topic |
|---|----------|-------|
| 01 | basic_tutorial | Getting started: data, indicators, plots, trades |
| 02 | data_analysis | Analysis: indicators, trades, market profile, order book |
| 03 | technical_indicators | Full indicator catalogue |
| 04 | plotting | All visualization capabilities |
| 05 | reversal_charts | Reversal candles from atomic trades |
| 06 | tagging_and_backtesting | Strategy signals and backtesting |
| 07 | support_resistance_kmeans | S/R with K-Means clustering |
| 08-10 | ichimoku_* | Ichimoku analysis and backtesting |
| 11 | exchange_info | Exchange class: metadata, filters, fees |
| 12 | export_csv | CSV export/import |
| 13-15 | database_* | PostgreSQL/TimescaleDB integration |


Documentation
-------------

Full Sphinx documentation: https://nand0san.github.io/binpan_studio/


GitHub repo
-----------

https://github.com/nand0san/binpan_studio


Google Colab
------------

Google Colab is not available for the Binance API due to IP restrictions.

```
BinanceAPIException: APIError(code=0): Service unavailable from a restricted location...
```


Jupyter Import Troubleshooting
------------------------------

If you encounter import errors in Jupyter, install packages directly to the kernel:

```python
import sys
!{sys.executable} -m pip install binpan
```

License
-------

MIT
