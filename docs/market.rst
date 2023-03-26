Market Module
=============

This module can manage market data from Binance API.

To import this module:

.. code-block::

   from handlers import market

.. automodule:: handlers.market

Prices
------

.. autofunction:: get_last_price

.. autofunction:: get_prices_dic


Klines
------

.. autofunction:: get_candles_by_time_stamps

.. autofunction:: get_historical_candles

.. autofunction:: parse_candles_to_dataframe


Format Data
-----------

.. autofunction:: basic_dataframe

.. autofunction:: convert_to_numeric


Trades
------

.. autofunction:: get_last_agg_trades

.. autofunction:: get_aggregated_trades

.. autofunction:: get_historical_agg_trades

.. autofunction:: parse_agg_trades_to_dataframe

.. autofunction:: get_last_atomic_trades

.. autofunction:: get_atomic_trades

.. autofunction:: get_historical_atomic_trades

.. autofunction:: parse_atomic_trades_to_dataframe

Orderbook
---------

.. autofunction:: get_order_book


Coin Conversion
---------------

.. autofunction:: intermediate_conversion

.. autofunction:: convert_coin

