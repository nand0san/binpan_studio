from .time_helper import convert_utc_ms_column_to_time_zone, convert_datetime_to_string, convert_milliseconds_to_utc_string, \
    convert_milliseconds_to_time_zone_datetime

import pandas as pd
import json

##################
# Redis Requests #
##################


def fetch_keys(redisClient: object,
               filter_tick_interval=None):
    """
    Fetch all keys in redis database.

    :return list: Returns all keys for a tick interval if passed, else all existing keys in redis.
    """
    ret = redisClient.scan_iter()
    if filter_tick_interval:
        ret = [i for i in ret if i.endswith(filter_tick_interval)]
    return ret


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


def redis_parser(symbol_json_data: list,
                 symbol: str,
                 tick_interval: str,
                 time_zone: str = 'UTC',
                 time_index: bool = True,
                 ) -> pd.DataFrame:
    """
    Parses redis klines list to a BinPan Dataframe.

    :param list symbol_json_data: A list with klines format data.
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
    dicts_data = [json.loads(i) for i in symbol_json_data]
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


def fetch_filter_query(redisClient: object,
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
    :param str symbol_filter: A symbol string to filter out all redis keys without it.
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
