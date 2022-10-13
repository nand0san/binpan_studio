Market Module
=============

This module can manage market data from Binance API.

To import this module:

.. code-block::

   from handlers import market

Klines
------

.. automodule:: handlers.market

.. autofunction:: get_candles_by_time_stamps

.. autofunction:: parse_candles_to_dataframe

Trades
------

.. autofunction:: get_agg_trades

.. autofunction:: get_historical_aggregated_trades

Coin Conversion
---------------

.. autofunction:: convert_coin

