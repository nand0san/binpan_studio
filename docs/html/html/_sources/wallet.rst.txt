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

.. autofunction:: get_fills_price

.. autofunction:: get_fills_qty

.. autofunction:: get_spot_trades_list

.. autofunction:: get_margin_trades_list


Spot Balances
-------------

.. autofunction:: get_spot_account_info

.. autofunction:: get_spot_free_balances

.. autofunction:: get_spot_locked_balances

.. autofunction:: get_coins_with_balance

.. autofunction:: get_spot_balances_df

.. autofunction:: get_spot_balances_total_value


Margin Balances
---------------

.. autofunction:: get_margin_account_details

.. autofunction:: get_margin_balances

.. autofunction:: get_margin_free_balances

.. autofunction:: get_margin_borrowed_balances

.. autofunction:: get_margin_interest_balances

.. autofunction:: get_margin_netAsset_balances

.. autofunction:: get_margin_balances_total_value


Helper
------

.. autofunction:: convert_str_date_to_ms







