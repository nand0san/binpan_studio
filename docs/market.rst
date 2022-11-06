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

.. autofunction:: parse_candles_to_dataframe


Format Data
-----------

.. autofunction:: basic_dataframe

.. autofunction:: convert_to_numeric


Trades
------

.. autofunction:: get_agg_trades

.. autofunction:: get_historical_aggregated_trades

.. autofunction:: parse_agg_trades_to_dataframe

.. autofunction:: get_last_trades

.. autofunction:: get_trades

Orderbook
---------

.. autofunction:: get_order_book


Coin Conversion
---------------

.. autofunction:: intermediate_conversion

.. autofunction:: convert_coin

