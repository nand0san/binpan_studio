Wallet Module
=============

This module can manage wallet data from Binance API.

To import this module:

.. code-block::

   from handlers import wallet

.. automodule:: handlers.wallet

Wallet snapshots
----------------

.. autofunction:: daily_account_snapshot

.. autofunction:: assets_convertible_dust

Trades
------

.. autofunction:: get_spot_trades_list

Balances
--------

.. autofunction:: get_spot_account_info

.. autofunction:: spot_free_balances_parsed

.. autofunction:: spot_locked_balances_parsed

.. autofunction:: get_coins_with_balance

.. autofunction:: get_spot_balances_df

.. autofunction:: get_spot_balances_total_value


Helper
------

.. autofunction:: convert_str_date_to_ms







