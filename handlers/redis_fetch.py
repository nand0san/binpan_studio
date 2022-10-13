from typing import List
from .time_helper import convert_utc_ms_column_to_time_zone, convert_datetime_to_string, convert_milliseconds_to_utc_string, \
    convert_milliseconds_to_time_zone_datetime, convert_milliseconds_to_str, open_from_milliseconds, next_open_by_milliseconds
from .market import tick_seconds
import pandas as pd
import json
from .logs import Logs
from redis import StrictRedis
from time import sleep, time
import numpy as np

redis_logger = Logs(filename='./logs/redis_fetch.log', name='redis_fetch', info_level='INFO')

klines_columns = {"t": "Open timestamp",
                  "o": "Open",
                  "h": "High",
                  "l": "Low",
                  "c": "Close",
                  "v": "Volume",
                  "T": "Close timestamp",
                  "q": "Quote volume",
                  "n": "Trades",
                  "V": "Taker buy base volume",
                  "Q": "Taker buy quote volume",
                  "B": "Ignore"}


##########
# Parser #
##########

def redis_klines_parser(json_list: List[str],
                        symbol: str,
                        tick_interval: str,
                        time_zone: str = 'Europe/madrid',
                        time_index: bool = True,
                        ) -> pd.DataFrame:
    """
    Parses redis klines list to a BinPan Dataframe.

    :param List[str] json_list: A list with klines format data.
    :param str symbol: Symbol expected. Just for naming index.
    :param str tick_interval: Tick interval expected. Just for naming index.
    :param str time_zone:  A time zone for converting the index. Example: 'Europe/Madrid'
    :param bool time_index: If true, index are datetime format, else integer index.
    :return pd.DataFrame: A BinPan dataframe.

    """

    time_cols = ['Open time', 'Close time']
    dicts_data = [json.loads(i) for i in json_list]
    df = pd.DataFrame(data=dicts_data)
    df.rename(columns=klines_columns, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(arg=df[col], downcast='integer')

    df.loc[:, 'Open time'] = df['Open timestamp']
    df.loc[:, 'Close time'] = df['Close timestamp']

    if time_zone != 'UTC':  # converts to time zone the time columns
        for col in time_cols:
            df.loc[:, col] = convert_utc_ms_column_to_time_zone(df, col, time_zone=time_zone)
            df.loc[:, col] = df[col].apply(lambda x: convert_datetime_to_string(x))
    else:
        for col in time_cols:
            df.loc[:, col] = df[col].apply(lambda x: convert_milliseconds_to_utc_string(x))

    if time_index:
        date_index = df['Open timestamp'].apply(convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
        df.set_index(date_index, inplace=True)

    index_name = f"{symbol.upper()} {tick_interval} {time_zone}"
    df.index.name = index_name
    return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote volume', 'Trades', 'Taker buy base volume',
               'Taker buy quote volume', 'Ignore', 'Open timestamp', 'Close timestamp']]


###############
# Redis Utils #
###############

def fetch_keys(redisClient: StrictRedis,
               filter_tick_interval=None,
               filter_quote: str = None) -> list:
    """
    Fetch all keys in redis database.

    :param StrictRedis redisClient: A redis connector.
    :param str filter_tick_interval: Optional. A binance klines tick interval to fetch all keys for that interval.
    :param str filter_quote: Filter symbols without a quote.
    :return list: Returns all keys for a tick interval if passed, else all existing keys in redis.
    """
    ret = redisClient.scan_iter()
    if filter_tick_interval:
        ret = [i for i in ret if i.endswith(filter_tick_interval)]
    if filter_quote:
        ret = [i for i in ret if i.split('@')[0].endswith(filter_quote.lower())]
    return list(ret)


def klines_continuity(klines: list, tick_interval: str):
    """
    Check klines missing.

    :param list klines: List of raw candles response in a list of strings, not json parsed yet.
    :param str tick_interval: Binance klines interval.
    :return bool: If true, klines are continuous.
    """
    try:
        start = int(klines[0][6:19])
        end = int(klines[-1][6:19])
        ticks = int((end - start) / (tick_seconds[tick_interval] * 1000))
        assert ticks + 1 == len(klines)
        return True
    except:
        print("Klines not continuously spaced!")
        return False


def klines_ohlc_to_numpy(klines: list) -> tuple:
    """
    Extract open, high, low and close numpy arrays from raw klines (list of json strings).

    :param list klines: A list with json strings.
    :return tuple: A tuple with numpy arrays with data: open, high, low and close.
    """
    open_ = []
    high = []
    low = []
    close = []
    for row in klines:
        json_row = json.loads(row)

        open_.append(json_row['o'])
        high.append(json_row['h'])
        low.append(json_row['l'])
        close.append(json_row['c'])

    ret1 = np.fromiter(open_, dtype=np.float64)
    ret2 = np.fromiter(high, dtype=np.float64)
    ret3 = np.fromiter(low, dtype=np.float64)
    ret4 = np.fromiter(close, dtype=np.float64)

    return ret1, ret2, ret3, ret4


def redis_baliza(redis_client: StrictRedis,
                 symbol: str,
                 tick_interval: str,
                 time_zone='Europe/madrid') -> int:
    """
    Returns when Redis data available waiting until the next open timestamp appears.

    Expected format for the redis data is:

    .. code-block::

       '{"t": 1657651320000, "o": "0.10100000", "h": "0.10100000", "l": "0.10100000", "c": "0.10100000", "v": "0.00000000",
         "T": 1657651379999, "q": "0.00000000", "n": 0, "V": "0.00000000", "Q": "0.00000000", "B": "0"}'


    :param StrictRedis redis_client: A redis client.
    :param str symbol: Symbol.
    :param str tick_interval: Binance tick interval.
    :param str time_zone: A time zone for messages.
    :return int: Timestamp of the open waited for.
    """

    now = int(time() * 1000)
    my_open = open_from_milliseconds(now, tick_interval=tick_interval)
    next_open = next_open_by_milliseconds(now, tick_interval=tick_interval)

    redis_logger.debug(
        f"Baliza {symbol} {tick_interval} started at {convert_milliseconds_to_str(ms=my_open, timezoned=time_zone)} and waiting "
        f"until {convert_milliseconds_to_str(ms=next_open, timezoned=time_zone)} is open.")

    while True:

        now = int(time() * 1000)
        last_redis_ts = fetch_zset_range(redisClient=redis_client,
                                         symbol=symbol.lower(),
                                         tick_interval=tick_interval,
                                         start_index=-1)
        fetched_open_ts = int(last_redis_ts[-1][6:19])

        # redis_logger.debug(f"Fetched {convert_milliseconds_to_str(ms=fetched_open_ts, timezoned=time_zone)}")

        if fetched_open_ts >= my_open:  # because redis candles apper just when closed

            redis_logger.debug(
                f"Fetched at {convert_milliseconds_to_str(ms=now, timezoned=time_zone)} with a late of {(now - next_open) / 1000} seconds.")
            break

        else:
            sleep(0.01)

    return fetched_open_ts


###############
# Redis Zsets #
###############

def push_to_ordered_set(redisClient: StrictRedis,
                        key: str,
                        mapping: dict,
                        LT=False,
                        XX=False,
                        NX=True,
                        GT=False,
                        CH=True,
                        INCR=False,
                        ):
    """
    Pushes elements to the ordered set with a score as index.

    :param StrictRedis redisClient: A redis client connector.
    :param str key: The redis key name.
    :param list mapping: Data to push in the format {data: score}
    :param LT: Only update existing elements if the new score is less than the current score. This flag doesn't prevent adding new elements.
    :param XX: Only update elements that already exist. Don't add new elements.
    :param NX: Only add new elements. Don't update already existing elements. This is the default only true value.
    :param GT: Only update existing elements if the new score is greater than the current score. This flag doesn't prevent adding new
       elements.
    :param CH: Modify the return value from the number of new elements added, to the total number of elements changed (CH is an abbreviation
       of changed). Changed elements are new elements added and elements already existing for which the score was updated. So elements
       specified in the command line having the same score as they had in the past are not counted. Note: normally the return value of ZADD
       only counts the number of new elements added.
    :param INCR: When this option is specified ZADD acts like ZINCRBY. Only one score-element pair can be specified in this mode.
       Note: The GT, LT and NX options are mutually exclusive.
    :return: redis feedback info.

    """
    return redisClient.zadd(name=key, mapping=mapping, lt=LT, xx=XX, nx=NX, gt=GT, ch=CH, incr=INCR)


def fetch_zset_range(redisClient: StrictRedis,
                     key: str = None,
                     symbol=None,
                     tick_interval=None,
                     start_index=0,
                     end_index=-1,
                     with_scores=False) -> list:
    """
    Fetch a redis ordered set by its key name and start end indexes. Optionally returns with redis index (scores) in a tuple each row
    if with_Scores.

    :param StrictRedis redisClient: A redis connector.
    :param str key: Redis key name. Optional, symbol and tick interval can be used too.
    :param str symbol: A Binance valid Symbol.
    :param str tick_interval: A binance tick interval. Like 5m, 1h etc
    :param int start_index: Numeric index in redis key.
    :param int end_index: Numeric index in redis key.
    :param bool with_scores: Will return rows with its index in a tuple if true.
    :return list: A list of rows in the redis set or a list of tuples with rows and scores if with_scores true.


    Example:

    .. code-block::

        from binpan import binpan
        from handlers import redis_fetch
        from random import choice

        redis_client = binpan.redis_client(ip='192.168.69.43')

        keys = redis_fetch.fetch_keys(redisClient=redis_client)
        stream = choice(keys)

        print(stream)

        >>> 'galabusd@kline_5m'

        redis_fetch.fetch_zset_range(redisClient=redis_client, key=stream, start_index=0, end_index=3)

        >>> ['{"t": 1658765700000, "o": "0.05011000", "h": "0.05017000", "l": "0.05004000", "c": "0.05006000", "v": "345741.00000000", "T": 1658765999999, "q": "17315.68030000", "n": 87, "V": "84642.00000000", "Q": "4239.51131000", "B": "0"}',
             '{"t": 1658766000000, "o": "0.05005000", "h": "0.05010000", "l": "0.04999000", "c": "0.04999000", "v": "448270.00000000", "T": 1658766299999, "q": "22422.46695000", "n": 68, "V": "132503.00000000", "Q": "6628.86299000", "B": "0"}',
             '{"t": 1658766300000, "o": "0.04998000", "h": "0.05004000", "l": "0.04995000", "c": "0.05000000", "v": "268084.00000000", "T": 1658766599999, "q": "13396.55626000", "n": 57, "V": "94341.00000000", "Q": "4715.14561000", "B": "0"}',
             '{"t": 1658766600000, "o": "0.04998000", "h": "0.05010000", "l": "0.04996000", "c": "0.05002000", "v": "402674.00000000", "T": 1658766899999, "q": "20162.97689000", "n": 68, "V": "154895.00000000", "Q": "7753.09416000", "B": "0"}']

    """
    if not key:
        key = f"{symbol}@kline_{tick_interval}"
    return redisClient.zrange(name=key, start=start_index, end=end_index, withscores=with_scores)


def fetch_zset_timestamps(redisClient: StrictRedis,
                          key: str,
                          start_timestamp: int,
                          end_timestamp: int = None,
                          with_scores=False) -> list:
    """
    Fetch a redis ordered set by its key name and start end scores. Optionally with redis index (scores) in a tuple each row.
    Optionally returns with redis index (scores) in a tuple each row if with_Scores.

    :param StrictRedis redisClient: A redis connector.
    :param str key: Redis key name.
    :param int start_timestamp: Usually a timestamp as index in redis key.
    :param int end_timestamp: Usually a timestamp as index in redis key.
    :param bool with_scores: Will return rows with its index in a tuple if true.
    :return list: A list of rows in the redis set or a list of tuples with rows and scores if with_scores true.

    Example:

    .. code-block::

        from binpan import binpan
        from handlers import redis_fetch
        from random import choice

        redis_client = binpan.redis_client(ip='192.168.69.43')

        keys = redis_fetch.fetch_keys(redisClient=redis_client)
        stream = choice(keys)

        print(stream)

        >>> 'galabusd@kline_5m'

        redis_fetch.fetch_zset_timestamps(redisClient=redis_client, key=stream, start_timestamp=1658765700000, end_timestamp= 1658766600000, with_scores=True)

        >>> [('{"t": 1658765700000, "o": "0.05011000", "h": "0.05017000", "l": "0.05004000", "c": "0.05006000", "v": "345741.00000000", "T": 1658765999999, "q": "17315.68030000", "n": 87, "V": "84642.00000000", "Q": "4239.51131000", "B": "0"}',
              1658765700000.0),
             ('{"t": 1658766000000, "o": "0.05005000", "h": "0.05010000", "l": "0.04999000", "c": "0.04999000", "v": "448270.00000000", "T": 1658766299999, "q": "22422.46695000", "n": 68, "V": "132503.00000000", "Q": "6628.86299000", "B": "0"}',
              1658766000000.0),
             ('{"t": 1658766300000, "o": "0.04998000", "h": "0.05004000", "l": "0.04995000", "c": "0.05000000", "v": "268084.00000000", "T": 1658766599999, "q": "13396.55626000", "n": 57, "V": "94341.00000000", "Q": "4715.14561000", "B": "0"}',
              1658766300000.0),
             ('{"t": 1658766600000, "o": "0.04998000", "h": "0.05010000", "l": "0.04996000", "c": "0.05002000", "v": "402674.00000000", "T": 1658766899999, "q": "20162.97689000", "n": 68, "V": "154895.00000000", "Q": "7753.09416000", "B": "0"}',
              1658766600000.0)]

        # Now without scores

        redis_fetch.fetch_zset_timestamps(redisClient=redis_client, key=stream, start_timestamp=1658765700000, end_timestamp= 1658766600000, with_scores=False)

        >>> ['{"t": 1658765700000, "o": "0.05011000", "h": "0.05017000", "l": "0.05004000", "c": "0.05006000", "v": "345741.00000000", "T": 1658765999999, "q": "17315.68030000", "n": 87, "V": "84642.00000000", "Q": "4239.51131000", "B": "0"}',
            '{"t": 1658766000000, "o": "0.05005000", "h": "0.05010000", "l": "0.04999000", "c": "0.04999000", "v": "448270.00000000", "T": 1658766299999, "q": "22422.46695000", "n": 68, "V": "132503.00000000", "Q": "6628.86299000", "B": "0"}',
            '{"t": 1658766300000, "o": "0.04998000", "h": "0.05004000", "l": "0.04995000", "c": "0.05000000", "v": "268084.00000000", "T": 1658766599999, "q": "13396.55626000", "n": 57, "V": "94341.00000000", "Q": "4715.14561000", "B": "0"}',
            '{"t": 1658766600000, "o": "0.04998000", "h": "0.05010000", "l": "0.04996000", "c": "0.05002000", "v": "402674.00000000", "T": 1658766899999, "q": "20162.97689000", "n": 68, "V": "154895.00000000", "Q": "7753.09416000", "B": "0"}']

    """
    ret = redisClient.zrangebyscore(name=key,
                                    min=start_timestamp,
                                    max=end_timestamp,
                                    withscores=with_scores)
    return ret


def fetch_set_and_parse(redisClient: StrictRedis,
                        key: str) -> pd.DataFrame:
    """
    Fetch a websocket channel from redis and parses to a BinPan dataframe.

    Data is expected to be in a redis ordered set with klines data each row and with Open timestamp as score.
    Example of data in one score of the zset:

    .. code-block::

       '{"t": 1657651320000, "o": "0.10100000", "h": "0.10100000", "l": "0.10100000", "c": "0.10100000", "v": "0.00000000",
         "T": 1657651379999, "q": "0.00000000", "n": 0, "V": "0.00000000", "Q": "0.00000000", "B": "0"}'

       # score for this entry would be Open timestamp: 1657651320000

    :param StrictRedis redisClient: A redis connector.
    :param str key: Key redis name.
    :return pd.DataFrame: BinPan dataframe.
    """
    symbol = key.split('@')[0].upper()
    tick_interval = key[-2:]
    data = fetch_zset_range(redisClient=redisClient, key=key.lower(), with_scores=False)
    if not data:
        redis_logger.warning(f"BinPan warning: No data found in Redis for {key}")
    return redis_klines_parser(json_list=data, symbol=symbol, tick_interval=tick_interval)


def zset_length_between_scores(redisClient: StrictRedis,
                               name: str,
                               min_index: int,
                               max_index: int) -> int:
    """
    Returns count of elements between start score and end score (including elements with score equal to min or max).

    :param StrictRedis redisClient: A redis client.
    :param str name: A redis available key (zset name).
    :param min_index: A score presumed value.
    :param max_index: A score presumed value.
    :return int: Count of scores in redis between ()
    """
    return redisClient.zcount(name=name, min=min_index, max=max_index)


def zset_length(redisClient: StrictRedis,
                name: str) -> int:
    """
    Returns count of elements in a set.

    :param StrictRedis redisClient: A redis client.
    :param str name: A redis available key (zset name).
    :return int: Count elements in zset.
    """
    return redisClient.zcard(name=name)


################
# List objects #
################


def push_line_to_redis(redisClient: StrictRedis,
                       key: str, data: str):
    """
    Pushes another line to the end of a list key in redis database.

    :param StrictRedis redisClient: A redis connector.
    :param str key: The redis key name.
    :param str data: Data to push. If not string type will be converted to string.
    :return: redis feedback info.

    """
    return redisClient.rpush(key, str(data))


def fetch_list(redisClient: StrictRedis,
               key: str,
               start_index=0,
               end_index=-1) -> list:
    """
    Fetch list entries in a redis list key from start index to end.

    :param object redisClient: A redis connector.
    :param str key: Key name in redis database.
    :param int start_index: First index to fetch.
    :param int end_index: Last index to fetch.
    :return list: All raw strings in redis list key into a python's list.

    """
    return redisClient.lrange(key, start_index, end_index)


def fetch_data_in_list(redisClient: StrictRedis,
                       key: str,
                       start_index=0,
                       end_index=-1) -> list:
    """
    Fetch data from redis for a given key. Data expected format is redis list.

    .. code-block::

       '{"t": 1657651320000, "o": "0.10100000", "h": "0.10100000", "l": "0.10100000", "c": "0.10100000", "v": "0.00000000",
         "T": 1657651379999, "q": "0.00000000", "n": 0, "V": "0.00000000", "Q": "0.00000000", "B": "0"}'


    :param object redisClient: A redis connector.
    :param str key: An existing redis key.
    :param int start_index: first index in the redis key to fetch for.
    :param int end_index: last index in the redis key to fetch for.
    :return:
    """
    return redisClient.lrange(key, start_index, end_index)


def insert_line_before_index_in_list(redisClient: StrictRedis,
                                     key: str,
                                     idx: int,
                                     value: str):
    """
    Inserts data before an existing string in a redis list. You need to know the index of the reference value.

    :param object redisClient: A redis connector.
    :param str key: A redis key list formatted.
    :param int idx: Index where will be inserted the value, pushing forward existing values, in other words,
        inserting before the referred value in that passed index.
    :param str value: An string to insert.
    :return None:

    """

    idx = int(idx)  # redis does not support numpy types
    old_value = redisClient.lindex(name=key, index=idx)
    tag = 'positional_tag'
    redisClient.lset(name=key, index=idx, value=tag)
    redisClient.linsert(name=key, where='AFTER', refvalue=tag, value=old_value)
    redisClient.linsert(name=key, where='AFTER', refvalue=tag, value=value)
    redisClient.lrem(name=key, count=0, value=tag)


def find_row_index_in_redis_key(redisClient: StrictRedis,
                                symbol: str,
                                tick_interval: str,
                                row_reference: str) -> int:
    """
    It founds in a redis stream the most accurate row with greater timestamp than the passed timestamp row and returns its
    index by iteration.

    :param object redisClient: A redis connector.
    :param str symbol: A binance symbol.
    :param str tick_interval: A binance klines tick interval.
    :param str row_reference: A json.dumps() of a row from binance klines.
    :return int: Index position of the next existing row.
    """
    stream = f"{symbol.lower()}@kline_{tick_interval}"
    my_row_ts = json.loads(row_reference)['t']

    # fetch redis klines full key, got to fetch because it does one by one change
    full_key_list = fetch_list(redisClient=redisClient,
                               key=stream,
                               start_index=0,
                               end_index=-1)
    key_jsons = [json.loads(i) for i in full_key_list]
    df = pd.DataFrame(key_jsons)
    df.loc[:, 'dif'] = df['t'] - my_row_ts
    greater = df[df['dif'] > 0].min()  # expected to get entire row
    if not greater.dropna().empty:
        greater_ts = greater['t']
        next_row = df[df['t'] == greater_ts]
        idx_to_insert = next_row.index[0]
        return idx_to_insert
    else:
        raise Exception("Not contemplated situation!!!!")


def fetch_list_filter_query(redisClient: StrictRedis,
                            coin_filter: str = None,
                            symbol_filter: str = None,
                            tick_interval_filter: str = None,
                            start_index=0,
                            end_index=-1) -> dict:
    """
    Fetch from redis all keys for a time interval. Then filters by symbol or coin in the ticker fetched.

    Is expected to fetch redis list data.

    .. code-block::

       '{"t": 1657651320000, "o": "0.10100000", "h": "0.10100000", "l": "0.10100000", "c": "0.10100000", "v": "0.00000000",
         "T": 1657651379999, "q": "0.00000000", "n": 0, "V": "0.00000000", "Q": "0.00000000", "B": "0"}'

    :param object redisClient: Connector for redis.
    :param str coin_filter: A coin string to filter out all redis keys without it.
    :param str symbol_filter: Optional. A symbol string to filter out all redis keys without it. Symbol filter prevales over coin filter.
    :param str tick_interval_filter: Binance klines tick interval.
    :param int start_index: first index in the redis key to fetch for.
    :param int end_index: last index in the redis key to fetch for.
    :return dict: A dict each symbol with its data.

    """

    if symbol_filter:
        symbol_filter = symbol_filter.lower()
    if coin_filter:
        coin_filter = coin_filter.lower()

    all_keys = fetch_keys(redisClient=redisClient,
                          filter_tick_interval=tick_interval_filter)
    if symbol_filter:
        all_keys = [i for i in all_keys if symbol_filter in i]
    elif coin_filter:
        all_keys = [i for i in all_keys if coin_filter in i]

    ret = {}
    for ticker in all_keys:
        data = fetch_data_in_list(redisClient=redisClient,
                                  key=ticker,
                                  start_index=start_index,
                                  end_index=end_index)
        ret[ticker] = data
    return ret


#############
# pipelines #
#############

# TODO: actualizar con lo nuevo de binance cache

def execute_pipeline(pipeline: StrictRedis.pipeline):
    """
    Executes a pipeline buffer.

    :param redisClient pipeline: A redis client pipeline.
    :return: Redis response.
    """
    return pipeline.execute()


def flush_pipeline(pipeline: StrictRedis.pipeline) -> StrictRedis.pipeline:
    """
    Flush pipeline buffered commands.

    :param redisClient pipeline: A redis client pipeline.
    :return: Redis response.
    """
    return pipeline.command_stack


def pipe_buffer_ordered_set(pipeline: StrictRedis,
                            key: str,
                            mapping: dict,
                            LT=False,
                            XX=False,
                            NX=True,
                            GT=False,
                            CH=True,
                            INCR=False,
                            ):
    """
        Fills a pipeline buffer to "pipeline.execute()" later with commands of pushes of elements to an ordered set with a score as index.

        :param redisClient pipeline: A redis client pipeline.
        :param str key: The redis key name.
        :param list mapping: Data to push in the format {data: score}
        :param LT: Only update existing elements if the new score is less than the current score. This flag doesn't prevent adding new elements.
        :param XX: Only update elements that already exist. Don't add new elements.
        :param NX: Only add new elements. Don't update already existing elements. This is the default only true value.
        :param GT: Only update existing elements if the new score is greater than the current score. This flag doesn't prevent adding new
           elements.
        :param CH: Modify the return value from the number of new elements added, to the total number of elements changed (CH is an abbreviation
           of changed). Changed elements are new elements added and elements already existing for which the score was updated. So elements
           specified in the command line having the same score as they had in the past are not counted. Note: normally the return value of ZADD
           only counts the number of new elements added.
        :param INCR: When this option is specified ZADD acts like ZINCRBY. Only one score-element pair can be specified in this mode.
           Note: The GT, LT and NX options are mutually exclusive.
        :return: redis feedback info.

    """
    pipeline.zadd(name=key, mapping=mapping, lt=LT, xx=XX, nx=NX, gt=GT, ch=CH, incr=INCR)
    return pipeline


def pipe_zset_range(pipeline: StrictRedis.pipeline,
                    key: str,
                    start_index=0,
                    end_index=-1,
                    with_scores=False) -> StrictRedis.pipeline:
    """
    Pipe a request for an ordered set by index for a redis existing key.

    :param StrictRedis.pipeline pipeline: Pipeline object.
    :param str key: Key to pipe wanted data.
    :param int start_index: Starting index of data. If ordered by scoring timestamps, starting values are older than ending values.
        Default 0
    :param int end_index: Ending index of data. If ordered by scoring timestamps, starting values are older than ending values.
        Default -1
    :param bool with_scores: Pipes for a list of tuples with data and score.
    :return StrictRedis.pipeline: Pipeline object.
    """
    return pipeline.zrange(name=key, start=start_index, end=end_index, withscores=with_scores)


def pipe_zset_timestamps(pipeline: StrictRedis.pipeline,
                         key: str,
                         start_timestamp: int,
                         end_timestamp: int = None,
                         with_scores=False) -> StrictRedis.pipeline:
    """
    Puts into pipeline buffer a redis ordered set by its key name and start end scores. Optionally with redis index (scores)
    in a tuple each row. Optionally returns with redis index (scores) in a tuple each row if with_Scores.

    :param StrictRedis.pipeline pipeline: A redis pipeline.
    :param str key: Redis key name.
    :param int start_timestamp: Usually a timestamp as index in redis key.
    :param int end_timestamp: Usually a timestamp as index in redis key.
    :param bool with_scores: Will return rows with its index in a tuple if true.
    :return StrictRedis.pipeline: A pipeline ready to execute.
    """
    return pipeline.zrangebyscore(name=key,
                                  min=start_timestamp,
                                  max=end_timestamp,
                                  withscores=with_scores)


def pipe_time_interval_bulk_ohlc_data(pipeline: StrictRedis.pipeline,
                                      min_index: int,
                                      max_index: int,
                                      my_keys: list,
                                      length=202,
                                      batch=200) -> dict:
    """
    Pipes to redis, check time integrity and parse in a list of tuples with stream and open, high, low, close arrays.
    """

    ohlc_arrays = {}

    for i in range(0, len(my_keys), batch):
        batch_keys = my_keys[i:i + batch]
        for key in batch_keys:
            pipeline = pipe_zset_timestamps(pipeline=pipeline,
                                            key=key,
                                            start_timestamp=min_index,
                                            end_timestamp=max_index)

        raw_klines = pipeline.execute()
        redis_logger.debug(f"Data to parse: {bool(any(raw_klines))} not empty len:{len([n for n in raw_klines if n])}")
        if not any(raw_klines):
            continue

        # check length
        for ii, k in enumerate(batch_keys):
            if len(raw_klines[ii]) != length:
                redis_logger.info(f"Missing klines for {k}")
                continue
            tick_interval = k.split('_')[-1]
            if klines_continuity(klines=raw_klines[ii], tick_interval=tick_interval):
                # parseo en numpy OHLC
                ohlc_arrays[k] = klines_ohlc_to_numpy(klines=raw_klines[ii])
            else:
                redis_logger.info(f"No continuity for {k}")

    if my_keys:
        redis_logger.debug(f"From {len(my_keys)} keys, valid: {len(ohlc_arrays)} - pct: {100 * len(ohlc_arrays) / len(my_keys):.2f}")
    else:
        redis_logger.info("No keys fetched!!!")
    return ohlc_arrays
