Trades Module
=============

The ``Trades`` value object wraps a trades DataFrame together with its metadata
(trade type, data origin and column map). It proxies attribute and item access to
the underlying DataFrame, so ``Symbol.agg_trades`` / ``Symbol.atomic_trades`` keep
behaving like DataFrames in notebooks (``s.agg_trades['Quantity']``, ``.empty``,
``.head()`` ...) while also exposing ``.df``, ``.trade_type``, ``.origin`` and helpers.

.. automodule:: binpan.core.trades
   :members:
   :undoc-members:
   :show-inheritance:
