Exchange Module
===============

This module manage exchange data from Binance API.

Can be imported this way:


.. code-block::

   from handlers import exchange

.. automodule:: handlers.exchange

Exchange General Data
---------------------

.. autofunction:: get_exchange_info


.. autofunction:: get_info_dic


Account
-------

.. autofunction:: get_account_status

Weight API control
------------------

.. autofunction:: get_exchange_limits


Symbols Trading Filters
-----------------------

.. autofunction:: get_symbols_filters

.. autofunction:: get_filters


Symbol Characteristics Filtering
--------------------------------

.. autofunction:: filter_tradeable

.. autofunction:: filter_spot

.. autofunction:: filter_margin

.. autofunction:: filter_not_margin

.. autofunction:: filter_leveraged_tokens

.. autofunction:: filter_legal

Symbol Operative Data
---------------------

.. autofunction:: get_precision

.. autofunction:: get_orderTypes_and_permissions

.. autofunction:: get_fees_dict

.. autofunction:: get_fees

.. autofunction:: get_system_status

.. autofunction:: get_coins_and_networks_info

.. autofunction:: get_coins_info_list

.. autofunction:: get_coins_info_dic

.. autofunction:: get_legal_coins

.. autofunction:: get_leveraged_coins

.. autofunction:: get_leveraged_symbols

.. autofunction:: get_quotes_dic

.. autofunction:: get_bases_dic

.. autofunction:: exchange_status

.. autofunction:: get_24h_statistics

.. autofunction:: not_iterative_coin_conversion

.. autofunction:: convert_to_other_coin

.. autofunction:: convert_symbol_base_to_other_coin

.. autofunction:: statistics_24h

.. autofunction:: get_top_gainers




