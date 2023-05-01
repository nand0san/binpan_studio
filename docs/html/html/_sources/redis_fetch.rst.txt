Redis Module
============

This module can manage Redis database data from Binance API with its original Binance format.

To import this module:

.. code-block::

   from handlers import redis_fetch

.. automodule:: handlers.redis_fetch

Redis recommended type is **Sorted Sets**. Check: https://redis.io/docs/manual/data-types/

Each element is expected to be indexed or **scored** (Redis call index number "score").

Example of an expected data response:

.. code-block::

    from binpan import binpan
    from handlers import redis_fetch
    from random import choice

    redis_client = binpan.redis_client(ip='192.168.69.43')

    keys = redis_fetch.fetch_keys(redisClient=redis_client)
    stream = choice(keys)

    print(stream)

    >>> 'galabusd@kline_5m'

    redis_fetch.fetch_zset_range(redisClient=redis_client, key=stream, start_index=0, end_index=3, with_scores=False)

    >>> ['{"t": 1658765700000, "o": "0.05011000", "h": "0.05017000", "l": "0.05004000", "c": "0.05006000", "v": "345741.00000000", "T": 1658765999999, "q": "17315.68030000", "n": 87, "V": "84642.00000000", "Q": "4239.51131000", "B": "0"}',
         '{"t": 1658766000000, "o": "0.05005000", "h": "0.05010000", "l": "0.04999000", "c": "0.04999000", "v": "448270.00000000", "T": 1658766299999, "q": "22422.46695000", "n": 68, "V": "132503.00000000", "Q": "6628.86299000", "B": "0"}',
         '{"t": 1658766300000, "o": "0.04998000", "h": "0.05004000", "l": "0.04995000", "c": "0.05000000", "v": "268084.00000000", "T": 1658766599999, "q": "13396.55626000", "n": 57, "V": "94341.00000000", "Q": "4715.14561000", "B": "0"}',
         '{"t": 1658766600000, "o": "0.04998000", "h": "0.05010000", "l": "0.04996000", "c": "0.05002000", "v": "402674.00000000", "T": 1658766899999, "q": "20162.97689000", "n": 68, "V": "154895.00000000", "Q": "7753.09416000", "B": "0"}']


Parsing
-------

.. autofunction:: redis_klines_parser

.. autofunction:: orderbook_value_to_dataframe


Utils
-----

.. autofunction:: fetch_keys

.. autofunction:: klines_continuity

.. autofunction:: klines_ohlc_to_numpy

.. autofunction:: redis_baliza


Fetching Redis Ordered Sets
---------------------------
It is expected to use as scoring, the "Open timestamp" or "timestamp", of each element in set.

.. autofunction:: push_to_ordered_set

.. autofunction:: fetch_zset_range

.. autofunction:: fetch_zset_timestamps

.. autofunction:: fetch_set_and_parse

.. autofunction:: zset_length_between_scores

.. autofunction:: zset_length


Fetching Redis lists
--------------------

.. autofunction:: push_line_to_redis

.. autofunction:: fetch_list

.. autofunction:: fetch_data_in_list

.. autofunction:: insert_line_before_index_in_list

.. autofunction:: find_row_index_in_redis_key

.. autofunction:: fetch_list_filter_query


Pipelines
---------

.. autofunction:: execute_pipeline

.. autofunction:: flush_pipeline

.. autofunction:: pipe_buffer_ordered_set

.. autofunction:: pipe_zset_timestamps

.. autofunction:: pipe_time_interval_bulk_ohlc_data








