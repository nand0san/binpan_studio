BinPan Module
=============

.. code-block:: python

   import binpan

.. automodule:: binpan


Symbol Class
------------

The main entry point in BinPan. Fetches candlestick (kline) data from the Binance API
and provides methods for technical indicators, plotting, trade analysis, and strategy
backtesting.

The ``Symbol`` class uses a mixin architecture:

- **IndicatorsMixin**: technical indicators (EMA, RSI, MACD, Bollinger Bands, etc.)
- **PlottingMixin**: candlestick charts, trade visualizations, order book plots
- **StrategyMixin**: tagging, cross detection, backtesting engine

Example notebooks are in the ``notebooks/`` folder of the repository.

.. image:: images/candles_ta.png
   :width: 1000
   :alt: Symbol chart with EMA and Supertrend indicators

.. autoclass:: Symbol
   :members:
   :inherited-members:


Exchange Class
--------------

Provides access to Binance exchange-level metadata: trading pairs, order types, filters,
fees, coin information, blockchain network details, and 24h volume rankings.

.. autoclass:: Exchange
   :members:


Wallet Class
------------

Shows wallet data: free and locked assets, wallet snapshots for performance analysis.
Requires API key and secret.

.. autoclass:: Wallet
   :members:


Database Class
--------------

Connector for PostgreSQL/TimescaleDB databases for storing and retrieving historical
market data (klines, trades).

.. autoclass:: Database
   :members:
