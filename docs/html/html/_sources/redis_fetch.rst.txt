Redis Module
============

This module can manage Redis database data from Binance API with its original Binance format.

.. automodule:: handlers.redis_fetch

Fetching Database Keys
----------------------

.. autofunction:: fetch_keys

Fetching Redis lists
--------------------

.. autofunction:: fetch_list

.. autofunction:: fetch_data_in_list

.. autofunction:: insert_line_before_index_in_list

.. autofunction:: find_row_index_in_redis_key

.. autofunction:: fetch_list_filter_query


Fetching Redis Ordered Sets
---------------------------
It is expected to use as scoring, the "Open timestamp" or timestamp, of each element in set.


.. autofunction:: push_to_ordered_set

.. autofunction:: fetch_zset_range

.. autofunction:: fetch_zset_timestamps

.. autofunction:: fetch_set_and_parse


