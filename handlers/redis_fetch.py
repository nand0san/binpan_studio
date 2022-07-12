from typing import List
from .time_helper import convert_utc_ms_column_to_time_zone, convert_datetime_to_string, convert_milliseconds_to_utc_string, \
    convert_milliseconds_to_time_zone_datetime
import pandas as pd
import json


##########
# Parser #
##########

def redis_klines_parser(json_list: List[str],
                        symbol: str,
                        tick_interval: str,
                        time_zone: str = 'UTC',
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


##################
# Redis Requests #
##################

def fetch_keys(redisClient: object,
               filter_tick_interval=None) -> list:
    """
    Fetch all keys in redis database.

    :param object redisClient: A redis connector.
    :param str filter_tick_interval: Optional. A binance klines tick interval to fetch all keys for that interval.
    :return list: Returns all keys for a tick interval if passed, else all existing keys in redis.
    """
    ret = redisClient.scan_iter()
    if filter_tick_interval:
        ret = [i for i in ret if i.endswith(filter_tick_interval)]
    return list(ret)


def push_line_to_redis(redisClient: object,
                       key: str, data: str):
    """
    Pushes another line to the end of a list key in redis database.

    :param object redisClient: A redis connector.
    :param str key: The redis key name.
    :param str data: Data to push. If not string type will be converted to string.
    :return: redis feedback info.

    """
    return redisClient.rpush(key, str(data))


def push_to_ordered_set(redisClient: object,
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

    :param object redisClient: A redis client connector.
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


def fetch_zset_range(redisClient: object,
                     key: str,
                     start_index=0,
                     end_index=-1,
                     with_scores=False) -> list:
    """
    Fetch a redis ordered set by its key name and start end indexes. Optionally returns with redis index (scores) in a tuple each row
    if with_Scores.

    :param object redisClient: A redis connector.
    :param str key: Redis key name.
    :param int start_index: Numeric index in redis key.
    :param int end_index: Numeric index in redis key.
    :param bool with_scores: Will return rows with its index in a tuple if true.
    :return list: A list of rows in the redis set or a list of tuples with rows and scores if with_scores true.
    """
    return redisClient.zrange(name=key, start=start_index, end=end_index, withscores=with_scores)


def fetch_zset_timestamps(redisClient: object,
                          key: str,
                          start_timestamp: int,
                          end_timestamp: int = None,
                          with_scores=False) -> list:
    """
    Fetch a redis ordered set by its key name and start end scores. Optionally with redis index (scores) in a tuple each row.
    Optionally returns with redis index (scores) in a tuple each row if with_Scores.

    :param object redisClient: A redis connector.
    :param str key: Redis key name.
    :param int start_timestamp: Usually a timestamp as index in redis key.
    :param int end_timestamp: Usually a timestamp as index in redis key.
    :param bool with_scores: Will return rows with its index in a tuple if true.
    :return list: A list of rows in the redis set or a list of tuples with rows and scores if with_scores true.
    """
    ret = redisClient.zrangebyscore(name=key,
                                    min=start_timestamp,
                                    max=end_timestamp,
                                    withscores=with_scores)
    return ret


def fetch_set_and_parse(redisClient: object,
                        stream: str) -> pd.DataFrame:
    """
    Fetch a websocket channel from redis and parses to a BinPan dataframe.

    Data is expected to be in a redis ordered set with klines data each row and with Open timestamp as score.
    Example of data in one score of the zset:

    .. code-block::

       '{"t": 1657651320000, "o": "0.10100000", "h": "0.10100000", "l": "0.10100000", "c": "0.10100000", "v": "0.00000000",
         "T": 1657651379999, "q": "0.00000000", "n": 0, "V": "0.00000000", "Q": "0.00000000", "B": "0"}'

       # score for this entry would be Open timestamp: 1657651320000

    :param object redisClient: A redis connector.
    :param str stream: Key redis name.
    :return pd.DataFrame: BinPan dataframe.
    """
    symbol = stream.split('@')[0].upper()
    tick_interval = stream[-2:]
    data = fetch_zset_range(redisClient=redisClient, key=stream, with_scores=False)
    return redis_klines_parser(json_list=data, symbol=symbol, tick_interval=tick_interval)


def fetch_list(redisClient: object,
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


def fetch_data_in_list(redisClient: object,
                       key: str,
                       start_index=0,
                       end_index=-1) -> list:
    """
    Fetch data from redis for a given key. Data expected format is redis list.

    :param object redisClient: A redis connector.
    :param str key: An existing redis key.
    :param int start_index: first index in the redis key to fetch for.
    :param int end_index: last index in the redis key to fetch for.
    :return:
    """
    return redisClient.lrange(key, start_index, end_index)


def insert_line_before_index_in_list(redisClient: object,
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


def find_row_index_in_redis_key(redisClient: object,
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


def fetch_list_filter_query(redisClient: object,
                            coin_filter: str = None,
                            symbol_filter: str = None,
                            tick_interval_filter: str = None,
                            start_index=0,
                            end_index=-1) -> dict:
    """
    Fetch from redis all keys for a time interval. Then filters by symbol or coin in the ticker fetched.

    Is expected to fetch redis list data.

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
