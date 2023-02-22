Aggregations Module
===================

This module can manage data aggregation.

This module can be imported:

.. code-block::

   from handlers import aggregations

.. automodule:: handlers.aggregations

Aggregations
------------

.. autofunction:: generate_count_grouper_column

.. autofunction:: ohlc_group

.. autofunction:: sum_split_by_boolean_column_and_group

.. autofunction:: drop_aggregated

.. autofunction:: tag_by_accumulation



DataFrame Tools
---------------

.. autofunction:: time_index_from_timestamps

.. autofunction:: columns_restriction

.. autofunction:: generate_volume_column


Concepts
--------

.. autofunction:: sign_of_price



AFML Methods
------------

This methods manage data following the Chapter 2 of the book Advances in Financial
Machine Learning by Macros López de Prado.

.. autofunction:: tick_bars

.. autofunction:: volume_bars

.. autofunction:: dollar_bars

.. autofunction:: imbalance_bars_divergent

.. autofunction:: imbalance_bars_fixed



