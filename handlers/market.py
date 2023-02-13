"""

Market functions

"""

from tqdm import tqdm
from time import time
import pandas as pd
import json
from decimal import Decimal as dd
from redis import StrictRedis
from typing import List

import handlers.time_helper
from .logs import Logs
from .quest import check_weight, get_response, api_raw_get, get_semi_signed_request
# from .time_helper import tick_seconds, end_time_from_start_time, start_time_from_end_time
from .time_helper import tick_seconds

market_logger = Logs(filename='./logs/market_logger.log', name='market_logger', info_level='INFO')

try:
    from secret import api_key
except ImportError:
    api_key, api_secret = "PLEASE ADD API KEY", "PLEASE ADD API SECRET"
    msg = """\n\n-------------------------------------------------------------
WARNING: No Binance API Key or API Secret. API key would be needed for personal API calls. Any other calls will work.

Adding example:

    from binpan import handlers
    
    handlers.files.add_api_key("xxxx")
    handlers.files.add_api_secret("xxxx")

API keys will be added to a file called secret.py in an encrypted way. API keys in memory stay encrypted except in the API call instant.

Create API keys: https://www.binance.com/en/support/faq/360002502072
"""
    market_logger.warning(msg)

base_url = 'https://api.binance.com'

klines_columns = {"t": "Open time",
                  "o": "Open",
                  "h": "High",
                  "l": "Low",
                  "c": "Close",
                  "v": "Volume",
                  "T": "Close time",
                  "q": "Quote volume",
                  "n": "Trades",
                  "V": "Taker buy base volume",
                  "Q": "Taker buy quote volume",
                  "B": "Ignore"}

trades_columns = {'M': 'Best price match',
                  'm': 'Buyer was maker',
                  'T': 'Timestamp',
                  'l': 'Last tradeId',
                  'f': 'First tradeId',
                  'q': 'Quantity',
                  'p': 'Price',
                  'a': 'Aggregate tradeId'}


##########
# Prices #
##########


def get_last_price(symbol: str = None) -> dict or list:
    """
    Returns all prices of symbols in a dict or if symbol specified, the float price of symbol.

    :param str symbol: A binance symbol.
    :return dict or list:

    """
    endpoint = '/api/v3/ticker/price'
    if symbol:
        weight = 1
        symbol = symbol.upper()
    else:
        weight = 2
    res = api_raw_get(endpoint=endpoint, weight=weight, params={'symbol': symbol})
    market_logger.debug(f"get_last_price: {res}")

    if type(res) == dict:
        return float(res['price'])
    elif type(res) == list:
        return {k['symbol']: float(k['price']) for k in res}


def get_prices_dic(decimal_mode: bool) -> dict:
    """
    Gets all symbols current prices into a dictionary.
    :decimal_mode bool: It flags to work in decimal mode.
    :return dict:
    """
    endpoint = '/api/v3/ticker/price'
    check_weight(2, endpoint=endpoint)
    ret = get_response(url=endpoint)
    if decimal_mode:
        return {d['symbol']: dd(d['price']) for d in ret}
    else:
        return {d['symbol']: float(d['price']) for d in ret}


###########
# Candles #
###########

def get_candles_by_time_stamps(symbol: str,
                               tick_interval: str,
                               start_time: int = None,
                               end_time: int = None,
                               limit=1000,
                               time_zone='Europe/Madrid',
                               redis_client: StrictRedis or dict = None) -> list:
    """
    Calls API for a candles list using one or two timestamps, starting and ending.

    In case the limit is passed and exceeded by requested time intervals, the start_time prevails over the end_time,
    start_time must come in milliseconds from epoch.

    In case of two timeStamps as arguments, limit is ignored.

    The API rounds the startTime up to the next open of the next candle. That is, it does not include the candle in which there is
    that timeStamp, but the next candle of the corresponding tick_interval, except in case it exactly matches the value of an open
    timestamp, in which case it will include it in the return.

    The indicated endTime will include the candlestick that timestamp is on. It should be in milliseconds. It can be a not closed one if
    is open right in between the endtime timestamp.

    If no timestamps are passed, the last quantity candlesticks up to limit count are returned.

    :param str symbol: A binance valid symbol.
    :param str tick_interval: A binance valid time interval for candlesticks.
    :param int start_time: A timestamp in milliseconds from epoch.
    :param int end_time: A timestamp in milliseconds from epoch.
    :param int limit: Count of candles to ask for.
    :param bool or StrictRedis or dict redis_client: A redis instance of a connector. Also can be passed a dictionary with redis client configuration. Example:

        redis_client = {'host': '192.168.89.242', 'port': 6379, 'db': 0, 'decode_responses': True}

    :param str time_zone: Just used for exception errors.
    :return list: Returns a list from the Binance API

    .. code-block::

        [
          [
            1499040000000,      // Open time
            "0.01634790",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore.
          ]
        ]

    """
    endpoint = '/api/v3/klines?'

    start_string = handlers.time_helper.convert_milliseconds_to_str(ms=start_time, timezoned=time_zone)
    end_string = handlers.time_helper.convert_milliseconds_to_str(ms=end_time, timezoned=time_zone)

    market_logger.debug(
        f"get_candles_by_time_stamps -> symbol={symbol} tick_interval={tick_interval} start={start_string} end={end_string}")

    tick_milliseconds = int(tick_seconds[tick_interval] * 1000)

    if not start_time and not end_time:
        end_time = int(time()) * 1000

    if not start_time and end_time:
        start_time = end_time - (limit * tick_milliseconds)
        start_time = handlers.time_helper.open_from_milliseconds(ms=start_time, tick_interval=tick_interval)

    elif start_time and not end_time:
        end_time = start_time + (limit * tick_milliseconds)  # ??? for getting limit exactly
        end_time = handlers.time_helper.open_from_milliseconds(ms=end_time, tick_interval=tick_interval)

    # prepare iteration for big loops
    tick_milliseconds = int(tick_seconds[tick_interval] * 1000)
    ranges = [(i, i + (1000 * tick_milliseconds)) for i in range(start_time, end_time, tick_milliseconds * 1000)]

    # loop
    raw_candles = []
    for r in ranges:
        start = r[0]
        end = r[1]

        if redis_client:
            if type(redis_client) == dict:
                try:
                    redis_client = StrictRedis(**redis_client)
                except Exception:
                    msg = f"BinPan exceptionRedis client error: arguments for client={redis_client}"
                    market_logger.error(msg)
                    raise Exception(msg)
            elif type(redis_client) != StrictRedis:
                msg = f"BinPan exceptionRedis client error: type passed={type(redis_client)}"
                market_logger.error(msg)
                raise Exception(msg)

            ret = redis_client.zrangebyscore(name=f"{symbol.lower()}@kline_{tick_interval}",
                                             min=start,
                                             max=end,
                                             withscores=False)
            if not ret:
                continue
            response = [json.loads(i) for i in ret]
        else:
            params = {'symbol': symbol,
                      'interval': tick_interval,
                      'startTime': start,
                      'endTime': end,
                      'limit': limit}
            params = {k: v for k, v in params.items() if v}
            check_weight(1, endpoint=endpoint)
            response = get_response(url=endpoint, params=params)

        raw_candles += response

    if not raw_candles:
        msg = f"BinPan Exception: Requested data for {symbol.lower()}@kline_{tick_interval} between {start_string} and " \
              f"{end_string} missing in redis server: ."
        market_logger.error(msg)
        raise Exception(msg)

    # descarta sobrantes
    overtime_candle_ts = handlers.time_helper.next_open_by_milliseconds(ms=end_time, tick_interval=tick_interval)
    if raw_candles:
        if type(raw_candles[0]) == list:  # if from binance
            raw_candles = [i for i in raw_candles if int(i[0]) < overtime_candle_ts]
        else:
            open_ts_key = list(raw_candles[0].keys())[0]
            raw_candles = [i for i in raw_candles if int(i[open_ts_key]) < overtime_candle_ts]

    return raw_candles


def parse_candles_to_dataframe(raw_response: list,
                               symbol: str,
                               tick_interval: str,
                               columns: list = None,
                               time_cols: list = None,
                               time_zone: str or None = 'UTC',
                               time_index=False) -> pd.DataFrame:
    """
    Format a list of lists by changing the indicated time fields to string format.

    Passing a time_zone, for example 'Europe/Madrid', will change the time from utc to the indicated zone.

    It will automatically sort the DataFrame using the first column of the time_cols list.

    The index of the DataFrame will be numeric correlative.

    :param list(lists) raw_response:        API klines response. List of lists.
    :param str symbol:          Symbol requested
    :param str tick_interval:   Tick interval between candles.
    :param list columns:         Column names. Default is BinPan dataframe columns.
    :param list time_cols:       Columns to take dates from.
    :param str or None time_zone: Optional. Time zone to convert dates in index.
    :param bool time_index:      True gets dates index, False just numeric index.
    :return:                    Pandas DataFrame

    """
    if not time_cols:
        time_cols = ['Open time', 'Close time']

    if not columns:
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote volume',
                   'Trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore']

    columns = [col for col in columns if not 'timestamp' in col]

    # check if redis columns
    if raw_response:
        if type(raw_response[0]) == list:
            df = pd.DataFrame(raw_response, columns=columns)
        else:
            response_keys = list(raw_response[0].keys())
            response_keys.sort()
            sort_columns = columns
            sort_columns.sort()

            if response_keys != sort_columns:  # json keys from redis different
                columns = list(klines_columns.keys())
                df = pd.DataFrame(raw_response, columns=columns)
                df.rename(columns=klines_columns, inplace=True)
            else:
                df = pd.DataFrame(raw_response, columns=columns)
    else:
        msg = f"BinPan Warning: No response to parse for {symbol} {tick_interval}. " \
              f"Check symbol requested time interval or {symbol} online status: binpan.Exchange().df.loc['{symbol}'].to_dict()['status']"
        market_logger.warning(msg)
        return pd.DataFrame(columns=columns + ['Open timestamp', 'Close timestamp'])

    # for col in df.columns:
    #     df[col] = pd.to_numeric(arg=df[col], downcast='integer')
    df = convert_to_numeric(data=df)

    df.loc[:, 'Open timestamp'] = df['Open time']
    df.loc[:, 'Close timestamp'] = df['Close time']

    if type(time_zone) == str and time_zone != 'UTC':  # converts to time zone the time columns
        for col in time_cols:
            df.loc[:, col] = handlers.time_helper.convert_utc_ms_column_to_time_zone(df, col, time_zone=time_zone)
            df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_datetime_to_string(x))
    else:
        for col in time_cols:
            df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_milliseconds_to_utc_string(x))

    if time_index and time_zone:
        date_index = df['Open timestamp'].apply(handlers.time_helper.convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
        df.set_index(date_index, inplace=True)
        index_name = f"{symbol} {tick_interval} {time_zone}"
        df.index.name = index_name
    else:
        index_name = f"{symbol} {tick_interval}"
        df.index.name = index_name

    # if came from a loop there are duplicated values in step connections to be removed.
    df.drop_duplicates(inplace=True)
    return df


###############
# format data #
###############

def basic_dataframe(data: pd.DataFrame,
                    exceptions: list = None,
                    actions_col='actions'
                    ) -> pd.DataFrame:
    """
    Delete all columns except: Open, High, Low, Close, Volume, actions.

    Some columns can be excepted.

    Useful to drop messed technical indicators columns in one shot.

    :param pd.DataFrame data:        A BinPan DataFrame
    :param list exceptions:  A list of columns to avoid dropping.
    :param str actions_col: A column with operative actions.
    :return pd.DataFrame: Pandas DataFrame

    """
    df_ = data.copy(deep=True)
    if actions_col not in df_.columns:
        if exceptions:
            return df_[['Open', 'High', 'Low', 'Close', 'Volume'] + exceptions].copy(deep=True)
        else:
            return df_[['Open', 'High', 'Low', 'Close', 'Volume']].copy(deep=True)
    else:
        if exceptions:
            return df_[['Open', 'High', 'Low', 'Close', 'Volume', actions_col] + exceptions].copy(deep=True)
        else:
            return df_[['Open', 'High', 'Low', 'Close', 'Volume', actions_col]].copy(deep=True)


def convert_to_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts to numeric all posible columns for a given Dataframe.

    :param pd.DataFrame data: A dataframe with columns
    :return pd.DataFrame: A dataframe with numeric values in each column that can be numeric.
    """

    for col in data.columns:
        try:
            data.loc[:, col] = pd.to_numeric(arg=data[col], downcast='integer', errors='raise')
        except Exception:
            pass
    return data


##########
# Trades #
##########


def get_agg_trades(symbol: str,
                   fromId: int = None,
                   toId: int = None,
                   limit=None,
                   startTime: int = None,
                   endTime: int = None,
                   previous_request=None,
                   redis_client_trades: StrictRedis = None) -> List[dict]:
    """
    Returns aggregated trades from id to limit or last trades if id not specified. Also is possible to get from starTime utc in
    milliseconds from epoch or until endtime milliseconds from epoch.

    Deprecated: If it is tested with more than 1 hour of trades, it gives error 1127 and if you adjust it to one hour,
    the maximum limit of 1000 is NOT applied.

    Update: API now limits to 1000 trades requests by id.

    Start time and end time not applied if trade id passed.

    Limit applied in fromId mode defaults to 500. Maximum is 1000.

    :param str symbol: A binance valid symbol.
    :param int fromId: An aggregated trade id.
    :param int toId: An aggregated trade id end, used just with redis.
    :param int limit: Count of trades to ask for.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param list previous_request: To avoid API limit, this function can be used recursively.
    :param bool redis_client_trades: A redis instance of a connector. Must be a trades redis connector, usually different configuration
     from candles redis server.
    :return list: Returns a list from the Binance API in dicts.

    .. code-block::

        [
          {
            "a": 26129,         // Aggregate tradeId
            "p": "0.01633102",  // Price
            "q": "4.70443515",  // Quantity
            "f": 27781,         // First tradeId
            "l": 27781,         // Last tradeId
            "T": 1498793709153, // Timestamp
            "m": true,          // Was the buyer the maker?
            "M": true           // Was the trade the best price match?
          }
        ]
    """
    if previous_request is None:
        previous_request = []

    endpoint = '/api/v3/aggTrades?'
    check_weight(1, endpoint=endpoint)

    if redis_client_trades:
        if fromId and limit:

            response = redis_client_trades.zrangebyscore(name=f"{symbol.lower()}@aggTrade",
                                                         min=fromId,
                                                         max=fromId + limit,
                                                         withscores=False)
        elif fromId and toId:
            response = redis_client_trades.zrangebyscore(name=f"{symbol.lower()}@aggTrade",
                                                         min=fromId,
                                                         max=toId,
                                                         withscores=False)

        elif startTime and endTime:

            response = []
            market_logger.info(f"Fetching all trades from redis server for {symbol}")

            curr_trades = redis_client_trades.zrange(name=f"{symbol.lower()}@aggTrade",
                                                     start=0,
                                                     end=-1,
                                                     withscores=True)

            curr_trades = [json.loads(i[0]) for i in curr_trades]
            market_logger.debug(f"Fetched {len(curr_trades)} aggregate trades for {symbol}")

            if curr_trades:
                timestamps_dict = {k['T']: k['a'] for k in curr_trades}
                trade_ids = [i for t, i in timestamps_dict.items() if startTime <= t <= endTime]
                market_logger.debug(f"List of trades {len(trade_ids)} for {symbol}")

                if trade_ids:
                    response = redis_client_trades.zrangebyscore(name=f"{symbol.lower()}@aggTrade",
                                                                 min=trade_ids[0],
                                                                 max=trade_ids[-1],
                                                                 withscores=False)
                    response = [json.loads(i) for i in response]
                else:
                    response = []

                market_logger.info(f"Clean aggregated {len(response)} trades found for {symbol}")

            if not response:
                start_str = handlers.time_helper.convert_milliseconds_to_utc_string(ms=startTime)
                end_str = handlers.time_helper.convert_milliseconds_to_utc_string(ms=endTime)
                market_logger.info(f"No trade IDs found for {symbol} for given interval {start_str} and {end_str} (UTC) in server.")

        else:
            market_logger.info(f"Request for trades from {symbol} returning ALL trades available in Redis.")

            ret_raw = redis_client_trades.zrange(name=f"{symbol.lower()}@aggTrade",
                                                 start=0,
                                                 end=-1,
                                                 withscores=False)
            response = []
            pbar = tqdm(ret_raw)
            for i in pbar:
                pbar.set_description(desc=f"Aggregate Trades for {symbol}", refresh=True)
                response.append(json.loads(i))

    else:  # API calls
        if fromId and not startTime and not endTime:
            query = {'symbol': symbol, 'limit': limit, 'fromId': fromId}

        elif startTime and endTime:  # Limited to one hour by api
            # if previous_request:
            #     print('Prev msg response')
            #     print(handlers.time_helper.convert_milliseconds_to_str(previous_request[0]['T'], timezoned='Europe/Madrid'))
            #     print(handlers.time_helper.convert_milliseconds_to_str(previous_request[-1]['T'], timezoned='Europe/Madrid'))
            # deprecated:
            # try:
            #     assert (endTime - startTime) <= (1000 * 60 * 60)
            # except AssertionError:
            #     msg = f"BinPan Error: Aggregate trades timestamps interval cannot be greater than 1 hour. Please use historical aggregate " \
            #           f"trades function: interval in minutes = {(endTime - startTime) / (1000 * 60)} start: {startTime} end: {endTime}"
            #     market_logger.error(msg)
            #     raise Exception(msg)

            query = {'symbol': symbol, 'limit': 1000, 'startTime': startTime}

            response = get_response(url=endpoint, params=query)
            response = previous_request + response
            last_timestamp = response[-1]['T']
            # print('Request result')
            # print(handlers.time_helper.convert_milliseconds_to_str(startTime, timezoned='Europe/Madrid'))
            # print(handlers.time_helper.convert_milliseconds_to_str(last_timestamp, timezoned='Europe/Madrid'))

            # dupes remove
            response_prev = set()
            unique_response = []
            for dic in response:
                t = tuple(dic.items())
                if t not in response_prev:
                    response_prev.add(t)
                    unique_response.append(dic)

            if last_timestamp >= endTime:
                return sorted(response, key=lambda x: x['T'])
            else:
                market_logger.info(f"Missing timestamps yet: {endTime}-{last_timestamp} = {endTime - last_timestamp}")
                ret = sorted(response, key=lambda x: x['T'])
                return get_agg_trades(symbol=symbol, startTime=last_timestamp, endTime=endTime, previous_request=ret)

        elif startTime or endTime:
            # query = {'symbol': symbol, 'limit': limit}
            query = {'symbol': symbol, 'limit': limit, 'startTime': startTime, 'endTime': endTime}

            # deprecated:
            # if startTime and not endTime:
            #     query.update({'startTime': startTime})
            #     query.update({'endTime': end_time_from_start_time(startTime=startTime,
            #                                                       limit=1000,
            #                                                       tick_interval='1h')})
            # if endTime and not startTime:
            #     query.update({'endTime': endTime})
            #     query.update({'startTime': start_time_from_end_time(endTime=endTime,
            #                                                         limit=1000,
            #                                                         tick_interval='1h')})
        else:  # last ones
            query = {'symbol': symbol, 'limit': limit}

        response = get_response(url=endpoint, params=query)

    return response


# DEPRECATED
# def get_historical_aggregated_trades(symbol: str,
#                                      startTime: int,
#                                      endTime: int,
#                                      redis_client_trades: StrictRedis = None) -> List[dict]:
#     """
#     Returns aggregated trades between timestamps. API changed limit for requesting aggregated trades, now it is 1000 trades each request.
#
#     :param int startTime: A timestamp in milliseconds from epoch.
#     :param int endTime: A timestamp in milliseconds from epoch.
#     :param str symbol: A binance valid symbol.
#     :param bool redis_client_trades: A redis instance of a connector. Must be a trades redis connector, usually different configuration
#      from candles redis server.
#     :return list: Returns a list from the Binance API
#
#     .. code-block::
#
#         [
#           {
#             "a": 26129,         // Aggregate tradeId
#             "p": "0.01633102",  // Price
#             "q": "4.70443515",  // Quantity
#             "f": 27781,         // First tradeId
#             "l": 27781,         // Last tradeId
#             "T": 1498793709153, // Timestamp
#             "m": true,          // Was the buyer the maker?
#             "M": true           // Was the trade the best price match?
#           }
#         ]
#     """
#     # API put 1000 limit, not hour limit anymore
#     hour_ms = 60 * 60 * 1000
#     if (endTime - startTime) <= hour_ms:
#         return get_agg_trades(symbol=symbol, startTime=startTime, endTime=endTime, redis_client_trades=redis_client_trades)
#     else:
#         response = []
#         if not redis_client_trades:
#             pbar = tqdm(range(startTime, endTime, hour_ms))
#             for i in pbar:
#                 pbar.set_description(desc=f"Historical Aggregate Trades for {symbol}", refresh=True)
#                 response += get_agg_trades(symbol=symbol, startTime=i, endTime=i + hour_ms, redis_client_trades=redis_client_trades)
#         else:
#             response = []
#             market_logger.info(f"Fetching all trades from redis server for {symbol}")
#
#             curr_trades = redis_client_trades.zrange(name=f"{symbol.lower()}@aggTrade",
#                                                      start=0,
#                                                      end=-1,
#                                                      withscores=True)
#
#             curr_trades = [json.loads(i[0]) for i in curr_trades]
#             market_logger.debug(f"Fetched {len(curr_trades)} aggregate trades for {symbol}")
#
#             if curr_trades:
#                 timestamps_dict = {k['T']: k['a'] for k in curr_trades}
#                 trade_ids = [i for t, i in timestamps_dict.items() if startTime <= t <= endTime]
#                 market_logger.debug(f"List of trades {len(trade_ids)} for {symbol}")
#
#                 if trade_ids:
#                     response = redis_client_trades.zrangebyscore(name=f"{symbol.lower()}@aggTrade",
#                                                                  min=trade_ids[0],
#                                                                  max=trade_ids[-1],
#                                                                  withscores=False)
#                     response = [json.loads(i) for i in response]
#                 else:
#                     response = []
#
#                 market_logger.info(f"Clean aggregated {len(response)} trades found for {symbol}")
#
#             if not response:
#                 start_str = handlers.time_helper.convert_milliseconds_to_utc_string(ms=startTime)
#                 end_str = handlers.time_helper.convert_milliseconds_to_utc_string(ms=endTime)
#                 market_logger.info(f"No trade IDs found for {symbol} for given interval {start_str} and {end_str} (UTC) in server.")
#         return response


def parse_agg_trades_to_dataframe(response: list,
                                  columns: dict,
                                  symbol: str,
                                  time_zone: str = None,
                                  time_index: bool = None):
    """
    Parses the API response into a pandas dataframe.

    .. code-block::

         {'M': True,
          'T': 1656166914571,
          'a': 1218761712,
          'f': 1424997754,
          'l': 1424997754,
          'm': True,
          'p': '21185.05000000',
          'q': '0.03395000'}

    :param list response: API raw response from trades.
    :param columns: Column names.
    :param symbol: The used symbol.
    :param time_zone: Selected time zone.
    :param time_index: Or integer index.
    :return: Pandas DataFrame

    """
    if not response:
        return pd.DataFrame(columns=list(columns.values()))

    df = pd.DataFrame(response)
    df.rename(columns=columns, inplace=True)
    # df.loc[:, 'Buyer was maker'] = df['Buyer was maker'].replace({'Maker buyer': 1, 'Taker buyer': 0})

    df = convert_to_numeric(data=df)

    timestamps_col = 'Timestamp'
    timestamps_serie = df[timestamps_col].copy()
    col = 'Date'
    df.loc[:, col] = df[timestamps_col]
    if time_zone != 'UTC':  # converts to time zone the time columns
        df.loc[:, col] = handlers.time_helper.convert_utc_ms_column_to_time_zone(df, col, time_zone=time_zone)
        df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_datetime_to_string(x))
    else:
        df.loc[:, col] = df[timestamps_col].apply(lambda x: handlers.time_helper.convert_milliseconds_to_utc_string(x))

    if time_index:
        date_index = timestamps_serie.apply(handlers.time_helper.convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
        df.set_index(date_index, inplace=True)

    index_name = f"{symbol} {time_zone}"
    df.index.name = index_name
    # df.loc[:, 'Buyer was maker'] = df['Buyer was maker'].astype(bool)
    return df[['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Date', 'Timestamp', 'Buyer was maker',
               'Best price match']]


def get_last_atomic_trades(symbol: str,
                           limit=1000):
    """
    Returns recent atomic (not aggregated) trades.

    GET /api/v3/trades

    Get recent trades.

    Weight(IP): 1

    :param str symbol: A binance valid symbol.
    :param int limit: API max limit is 1000.
    :return list: Returns a list from the Binance API

    Return example:

    .. code-block::

        [
          {
            "id": 28457,
            "price": "4.00000100",
            "qty": "12.00000000",
            "quoteQty": "48.000012",
            "time": 1499865549590,
            "isBuyerMaker": true,
            "isBestMatch": true
          }
        ]

    """

    endpoint = '/api/v3/trades?'
    check_weight(1, endpoint=endpoint)
    query = {'symbol': symbol, 'limit': limit}
    return get_response(url=endpoint, params=query)


def get_atomic_trades(symbol: str,
                      fromId: int = None,
                      limit: int = None,
                      decimal_mode: bool = False) -> list:
    """
    Returns atomic (not aggregated) trades from id to limit or last trades if id not specified.

    Limit applied in fromId mode defaults to 500. Maximum is 1000.

    GET /api/v3/historicalTrades

    Get older market trades.

    Weight(IP): 5

    :param int fromId: Trade id to fetch from. If not passed, gets most recent trades.
    :param str symbol: A binance valid symbol.
    :param int limit: Count of trades to ask for.
    :param bool decimal_mode: Enables decimal type for return.
    :return list: Returns a list from the Binance API.

    .. code-block::

        [
          {
            "id": 28457,
            "price": "4.00000100",
            "qty": "12.00000000",
            "quoteQty": "48.000012",
            "time": 1499865549590, // Trade executed timestamp, as same as `T` in the stream
            "isBuyerMaker": true,
            "isBestMatch": true
          }
        ]
    """

    endpoint = '/api/v3/historicalTrades?'
    check_weight(5, endpoint=endpoint)
    query = {'symbol': symbol, 'limit': limit, 'fromId': fromId, 'recWindow': None}
    return get_semi_signed_request(url=endpoint,
                                   decimal_mode=decimal_mode,
                                   api_key=api_key,
                                   params=query)


def get_historical_atomic_trades(symbol: str,
                                 startTime: int = None,
                                 endTime: int = None,
                                 start_trade_id: int = None,
                                 end_trade_id: int = None,
                                 limit: int = 1000,
                                 redis_client_trades: StrictRedis = None) -> List[dict]:
    """
    Returns atomic (not aggregated) trades between timestamps. It iterates over limit 1000 intervals to adjust to API limit.

    This request can be very slow because the API request weight limit.

    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int start_trade_id: A trade id as first one (older).
    :param int end_trade_id: A trade id as last one (newer).
    :param int limit: Limit for missing heads or tails of the interval requested with timestamps or trade ids.
    :param str symbol: A binance valid symbol.
    :param bool redis_client_trades: A redis instance of a connector. Must be a trades redis connector, usually different configuration
     from candles redis server.
    :return list: Returns a list from the Binance API

    .. code-block::

        Last Trades example:
            [
                {'id': 86206215,
                 'price': '0.00454100',
                 'qty': '0.02400000',
                 'quoteQty': '0.00010898',
                 'time': 1669579405932,
                 'isBuyerMaker': False,
                 'isBestMatch': True}, ...
             ]
    """

    if start_trade_id and not end_trade_id:
        end_trade_id = start_trade_id + limit
    elif end_trade_id and not start_trade_id:
        start_trade_id = end_trade_id - limit

    trades = handlers.market.get_last_atomic_trades(symbol=symbol, limit=1000)
    requests_cnt = 0

    # with trade ids, find start
    if start_trade_id and end_trade_id and not redis_client_trades:
        current_first_trade = trades[0]['id']
        # while trades[0]['id'] > start_trade_id:
        while current_first_trade > start_trade_id:
            requests_cnt += 1
            market_logger.info(f"Requests to API for atomic trades of {symbol}: {requests_cnt}")
            fetched_older_trades = handlers.market.get_atomic_trades(symbol=symbol,
                                                                     fromId=(current_first_trade - 1000),
                                                                     limit=1000)
            trades = fetched_older_trades + trades
            current_first_trade = trades[0]['id']

        current_last_trade = trades[-1]['id']
        while current_last_trade < end_trade_id:
            requests_cnt += 1
            market_logger.info(f"Requests to API for atomic trades of {symbol}: {requests_cnt}")
            fetched_older_trades = handlers.market.get_atomic_trades(symbol=symbol,
                                                                     fromId=current_last_trade,
                                                                     limit=1000)
            trades = fetched_older_trades + trades
            current_last_trade = trades[-1]['id']

        ret = [i for i in trades if start_trade_id <= i['id'] <= end_trade_id]
        return sorted(ret, key=lambda x: x['id'])

    # with timestamps or trade ids
    if not redis_client_trades and (startTime or endTime):
        current_first_trade_time = trades[0]['time']
        current_first_trade = trades[0]['id']
        # if startTime or endTime:
        if startTime:
            while current_first_trade_time >= startTime:
                requests_cnt += 1
                market_logger.info(f"Requests API for atomic trades STARTIME {symbol}: {requests_cnt}")
                fetched_older_trades = handlers.market.get_atomic_trades(symbol=symbol,
                                                                         fromId=current_first_trade - 1000,
                                                                         limit=1000)
                trades = fetched_older_trades + trades
                current_first_trade_time = trades[0]['time']
                current_first_trade = trades[0]['id']
        if endTime:
            current_last_trade = trades[-1]['id']
            current_last_trade_time = trades[-1]['time']
            while current_last_trade_time <= endTime:
                requests_cnt += 1
                market_logger.info(f"Requests API for atomic trades ENDTIME {symbol}: {requests_cnt}")
                fetched_newer_trades = handlers.market.get_atomic_trades(symbol=symbol,
                                                                         fromId=current_last_trade,
                                                                         limit=1000)
                trades += fetched_newer_trades
                current_last_trade = trades[-1]['id']
                current_last_trade_time = trades[-1]['time']

        ret = [i for i in trades if startTime <= i['time'] <= endTime]
        #
        # else:
        #     if start_trade_id:
        #         current_first_trade = trades[0]['id']
        #         while current_first_trade > start_trade_id:
        #             requests_cnt += 1
        #             market_logger.info(f"Requests API for atomic trades {symbol}: {requests_cnt}")
        #             fetched_older_trades = handlers.market.get_atomic_trades(symbol=symbol,
        #                                                                      fromId=(current_first_trade - 1000),
        #                                                                      limit=1000)
        #             trades = fetched_older_trades + trades
        #             current_first_trade = trades[0]['id']
        #
        #     if end_trade_id:
        #         current_last_trade = trades[-1]['id']
        #         while current_last_trade < end_trade_id:
        #             requests_cnt += 1
        #             market_logger.info(f"Requests API for atomic trades {symbol}: {requests_cnt}")
        #             fetched_newer_trades = handlers.market.get_atomic_trades(symbol=symbol,
        #                                                                      fromId=current_last_trade,
        #                                                                      limit=1000)
        #             trades += fetched_newer_trades
        #             current_last_trade = trades[-1]['id']
        #     ret = [i for i in trades if start_trade_id <= i['id'] <= end_trade_id]
        return sorted(ret, key=lambda x: x['id'])

    # from redis
    response = []
    market_logger.info(f"Fetching atomic trades from redis server for {symbol}")

    curr_trades = redis_client_trades.zrange(name=f"{symbol.lower()}@trade",
                                             start=0,
                                             end=-1,
                                             withscores=True)

    curr_trades = [json.loads(i[0]) for i in curr_trades]
    market_logger.debug(f"Fetched {len(curr_trades)} atomic trades for {symbol}")

    if curr_trades:
        timestamps_dict = {k['T']: k['t'] for k in curr_trades}
        trade_ids = [i for t, i in timestamps_dict.items() if startTime <= t <= endTime]
        market_logger.debug(f"List of trades {len(trade_ids)} for {symbol}")

        if trade_ids:
            response = redis_client_trades.zrangebyscore(name=f"{symbol.lower()}@trade",
                                                         min=trade_ids[0],
                                                         max=trade_ids[-1],
                                                         withscores=False)
            response = [json.loads(i) for i in response]
        else:
            response = []

        market_logger.info(f"Clean atomic {len(response)} trades found for {symbol}")

    if not response:
        start_str = handlers.time_helper.convert_milliseconds_to_utc_string(ms=startTime)
        end_str = handlers.time_helper.convert_milliseconds_to_utc_string(ms=endTime)
        market_logger.info(f"No atomic trade IDs found for {symbol} for given interval {start_str} and {end_str} (UTC) in server.")
    return response


def parse_atomic_trades_to_dataframe(response: list,
                                     columns: dict,
                                     symbol: str,
                                     time_zone: str = None,
                                     time_index: bool = None):
    """
    Parses the API response into a pandas dataframe.

    .. code-block::

        [
          {
            "id": 28457,
            "price": "4.00000100",
            "qty": "12.00000000",
            "quoteQty": "48.000012",
            "time": 1499865549590, // Trade executed timestamp, as same as `T` in the stream
            "isBuyerMaker": true,
            "isBestMatch": true
          }
        ]

        Redis response:

         {'e': 'trade',
          'E': 1669721069347,
          's': 'LTCBTC',
          't': 86293174,
          'p': '0.00466300',
          'q': '3.10900000',
          'b': 992385453,
          'a': 992385448,
          'T': 1669721069347,
          'm': False,
          'M': True},
         ...]

    :param list response: API raw response from atomic trades.
    :param columns: Column names.
    :param symbol: The used symbol.
    :param time_zone: Selected time zone.
    :param time_index: Or integer index.
    :return: Pandas DataFrame

    """
    if not response:
        return pd.DataFrame(columns=list(columns.values()))

    df = pd.DataFrame(response)
    df.rename(columns=columns, inplace=True)

    df = convert_to_numeric(data=df)

    timestamps_col = 'Timestamp'
    timestamps_serie = df[timestamps_col].copy()
    col = 'Date'
    df.loc[:, col] = df[timestamps_col]
    if time_zone != 'UTC':  # converts to time zone the time columns
        df.loc[:, col] = handlers.time_helper.convert_utc_ms_column_to_time_zone(df, col, time_zone=time_zone)
        df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_datetime_to_string(x))
    else:
        df.loc[:, col] = df[timestamps_col].apply(lambda x: handlers.time_helper.convert_milliseconds_to_utc_string(x))

    if time_index:
        date_index = timestamps_serie.apply(handlers.time_helper.convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
        df.set_index(date_index, inplace=True)

    index_name = f"{symbol} {time_zone}"
    df.index.name = index_name

    if 'quoteQty' in columns.keys():
        return df[['Trade Id', 'Price', 'Quantity', 'Quote quantity', 'Date', 'Timestamp', 'Buyer was maker', 'Best price match']]
    else:  # it was a redis response
        return df[['Trade Id', 'Price', 'Quantity', 'Buyer Order Id', 'Seller Order Id', 'Date', 'Timestamp', 'Buyer was maker',
                   'Best price match']]


#############
# Orderbook #
#############


def get_order_book(symbol: str, limit=5000) -> dict:
    """
    Returns a dictionary with timestamp, bids and asks. Bids and asks are a list of lists with strings representing price and quantity.

    :param str symbol: A Binance valid symbol.
    :param int limit: Max is 5000. Default 5000.
    :return dict: A dict with ask and bids

    .. code-block::

        {
          "lastUpdateId": 1027024,
          "bids": [
            [
              "4.00000000",     // PRICE
              "431.00000000"    // QTY
            ]
          ],
          "asks": [
            [
              "4.00000200",
              "12.00000000"
            ]
          ]
        }

    """

    endpoint = '/api/v3/depth?'
    check_weight(limit // 100 or 1, endpoint=endpoint)

    query = {'symbol': symbol, 'limit': limit}
    return get_response(url=endpoint, params=query)


#####################
# Order Book Ticker #
#####################

def get_orderbook_tickers(symbol: str = None, decimal_mode: bool = False) -> dict:
    """
    Symbol Order Book Ticker


    GET /api/v3/ticker/bookTicker

    Best price/qty on the order book for a symbol or symbols.

    Weight(IP):

    Parameter	Symbols Provided	Weight
    symbol	                    1	    1
    symbol parameter is omitted	        2
    symbols	Any	                        2
    Parameters:

    Name	Type	Mandatory	Description
    symbol	STRING	NO	Parameter symbol and symbols cannot be used in combination.

    If neither parameter is sent, bookTickers for all symbols will be returned in an array.

    Examples of accepted format for the symbols parameter: ["BTCUSDT","BNBUSDT"]

    :param str symbol: Optional. If not passed, all symbols returned.
    :param bool decimal_mode: If selected, return decimal values.

    :return: Api response is:

        .. code-block::
            Response:

            {
              "symbol": "LTCBTC",
              "bidPrice": "4.00000000",
              "bidQty": "431.00000000",
              "askPrice": "4.00000200",
              "askQty": "9.00000000"
            }
            OR

            [
              {
                "symbol": "LTCBTC",
                "bidPrice": "4.00000000",
                "bidQty": "431.00000000",
                "askPrice": "4.00000200",
                "askQty": "9.00000000"
              },
              {
                "symbol": "ETHBTC",
                "bidPrice": "0.07946700",
                "bidQty": "9.00000000",
                "askPrice": "100000.00000000",
                "askQty": "1000.00000000"
              }
            ]

    """
    endpoint = '/api/v3/ticker/bookTicker?'
    if symbol:
        weight = 1
    else:
        weight = 2
    check_weight(weight, endpoint=endpoint)
    query = {'symbol': symbol}
    response = get_response(url=endpoint, params=query)
    if type(response) == dict:
        ret = {response['symbol']: response}
    else:
        ret = {i['symbol']: i for i in response}
    if decimal_mode:
        ret = {k: {kk: dd(vv) for kk, vv in v.items() if kk != 'symbol'} for k, v in ret.items()}
    else:
        ret = {k: {kk: float(vv) for kk, vv in v.items() if kk != 'symbol'} for k, v in ret.items()}
    return ret


####################
# coin conversions #
####################


def intermediate_conversion(coin: str,
                            decimal_mode: bool,
                            prices: dict = None,
                            try_coin: str = 'BTC',
                            coin_qty: float = 1) -> float or None:
    """
    Uses an intermediate symbol for conversion. Uses stablecoin USDT versus other "try" coin.

    :param str coin: A binance coin.
    :param bool decimal_mode: It flags to work in decimal mode.
    :param dict prices: BinPan prices dict.
    :param str try_coin: Coin to try as intermediate in case of not existing pair. Default is BTC.
    :param float coin_qty: Quantity to convert. Default is 1.
    :return float: converted value.
    """

    if not prices:
        prices = get_prices_dic(decimal_mode=decimal_mode)

    if coin + try_coin in prices.keys():
        price = prices[coin + try_coin]
        try_symbol = f"{try_coin}USDT"
        return price * coin_qty * prices[try_symbol]

    elif try_coin + coin in prices.keys():
        price = 1 / prices[try_coin + coin]
        try_symbol = f"USDT{try_coin}"
        try:
            return price * coin_qty * prices[try_symbol]
        except KeyError:
            return None
    else:
        return None


def convert_coin(coin: str,
                 decimal_mode: bool,
                 convert_to: str = 'BUSD',
                 coin_qty: float or dd = 1,
                 prices: dict = None) -> float or None:
    """
    Calculates a coin quantity value converted to other coin with current exchange prices.

    :param str coin: An existing coin string.
    :param bool decimal_mode: It flags to work in decimal mode.
    :param str convert_to: An existing coin string.
    :param float coin_qty: How many coins to convert to.
    :param dict prices: A dictionary with symbols and prices.
    :return float: Converted value for the quantity
    """
    coin = coin.upper()
    convert_to = convert_to.upper()

    if coin == convert_to:
        return coin_qty

    if not prices:
        prices = get_prices_dic(decimal_mode=decimal_mode)

    symbol_a = coin + convert_to
    symbol_b = convert_to + coin

    if symbol_a in prices.keys():
        return coin_qty * prices[symbol_a]

    elif symbol_b in prices.keys():
        return coin_qty * (1 / prices[symbol_b])
    else:
        ret1 = intermediate_conversion(coin=coin, prices=prices, try_coin='BTC', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret1:
            return ret1
        ret2 = intermediate_conversion(coin=coin, prices=prices, try_coin='BUSD', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret2:
            return ret2
        ret3 = intermediate_conversion(coin=coin, prices=prices, try_coin='BNB', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret3:
            return ret3
        ret4 = intermediate_conversion(coin=coin, prices=prices, try_coin='ETH', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret4:
            return ret4
        ret5 = intermediate_conversion(coin=coin, prices=prices, try_coin='TUSD', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret5:
            return ret5
        ret6 = intermediate_conversion(coin=coin, prices=prices, try_coin='USDC', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret6:
            return ret6
        else:
            market_logger.warning(f"No possible conversion for {coin} to {convert_to}")
            return None
