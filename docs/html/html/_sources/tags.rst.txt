Tags & Backtesting Module
=========================

This module can tag values, compare and extract relations between indicators or any Serie of data. Also can backtest
any strategy.

To import this module:

.. code-block::

   from handlers import tags

.. automodule:: handlers.tags


Tags & Cross
------------

.. autofunction:: tag_value

.. autofunction:: tag_comparison

.. autofunction:: tag_cross

.. autofunction:: tag_column_to_strategy_group


Merge Series
------------

.. autofunction:: merge_series

.. autofunction:: clean_in_out


Backtesting
-----------

.. autofunction:: buy_base_backtesting

.. autofunction:: sell_base_backtesting

.. autofunction:: evaluate_wallets

.. autofunction:: check_action_labels_for_backtesting

.. autofunction:: simple_backtesting

.. autofunction:: backtesting

