"""

Market functions

"""

from time import time
import pandas as pd
import json
from decimal import Decimal as dd
from datetime import datetime
from typing import List

from .exceptions import BinPanException
from .logs import Logs
from .quest import check_weight, get_response, api_raw_get, get_semi_signed_request
from .time_helper import (tick_seconds, convert_milliseconds_to_str, convert_ms_column_to_datetime_with_zone,
                          convert_milliseconds_to_utc_string, convert_datetime_to_string, open_from_milliseconds, next_open_by_milliseconds,
                          convert_milliseconds_to_time_zone_datetime)
from .standards import (klines_columns, agg_trades_columns_from_binance, atomic_trades_columns_from_binance,
                        atomic_trades_columns_from_redis)
from .files import get_encoded_secrets

market_logger = Logs(filename='./logs/market_logger.log', name='market_logger', info_level='INFO')

base_url = 'https://api.binance.com'


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

    if type(res) is dict:
        return float(res['price'])
    elif type(res) is list:
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
                               time_zone='Europe/Madrid') -> list:
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

    start_string = convert_milliseconds_to_str(ms=start_time, timezoned=time_zone)
    end_string = convert_milliseconds_to_str(ms=end_time, timezoned=time_zone)

    chan = f"{symbol.lower()}@kline_{tick_interval}"
    market_logger.info(f"get_candles_by_time_stamps -> symbol={symbol} tick_interval={tick_interval} start={start_string} end={end_string} "
                       f"channel:{chan}")

    tick_milliseconds = int(tick_seconds[tick_interval] * 1000)

    if not start_time and not end_time:
        end_time = int(time()) * 1000

    if not start_time and end_time:
        start_time = end_time - (limit * tick_milliseconds)
        start_time = open_from_milliseconds(ms=start_time, tick_interval=tick_interval)

    elif start_time and not end_time:
        end_time = start_time + (limit * tick_milliseconds)  # ??? for getting limit exactly
        end_time = open_from_milliseconds(ms=end_time, tick_interval=tick_interval)

    # prepare iteration for big loops
    tick_milliseconds = int(tick_seconds[tick_interval] * 1000)
    ranges = [(i, i + (1000 * tick_milliseconds)) for i in range(start_time, end_time, tick_milliseconds * 1000)]

    # loop
    raw_candles = []
    for r in ranges:
        start = r[0]
        end = r[1]

        params = {'symbol': symbol, 'interval': tick_interval, 'startTime': start, 'endTime': end, 'limit': limit}
        params = {k: v for k, v in params.items() if v}
        check_weight(1, endpoint=endpoint)

        start_str = convert_milliseconds_to_str(start, timezoned=time_zone)
        end_str = convert_milliseconds_to_str(min(end, int(1000 * time())), timezoned=time_zone)
        market_logger.info(f"API request: {symbol} {start_str} to {end_str}")
        response = get_response(url=endpoint, params=params)

        raw_candles += response

    if not raw_candles:
        msg = f"BinPan Exception: Requested data for {symbol.lower()}@kline_{tick_interval} between {start_string} and " \
              f"{end_string} missing in API."
        market_logger.warning(msg)
        return []
        # raise Exception(msg)

    # descarta sobrantes
    overtime_candle_ts = next_open_by_milliseconds(ms=end_time, tick_interval=tick_interval)
    # if raw_candles:
    if type(raw_candles[0]) is list:  # if from binance
        raw_candles = [i for i in raw_candles if int(i[0]) < overtime_candle_ts]
    else:
        open_ts_key = list(raw_candles[0].keys())[0]
        raw_candles = [i for i in raw_candles if int(i[open_ts_key]) < overtime_candle_ts]

    return raw_candles


def get_historical_candles(symbol: str,
                           tick_interval: str,
                           start_time: int,
                           end_time: int,
                           tick_interval_ms: int,
                           limit: int = 1000,
                           ignore_errors: bool = False) -> list:
    """
    Retrieve all kline data within the given time range considering the API limit.

    Start time and end time are rounded down to the nearest open tick interval and are included both opens rounded.

    Please, check last close timestamp in klines because it can be not closed yet.

    :param str symbol: The trading pair symbol (e.g., "BTCUSDT").
    :param str tick_interval: Kline tick interval (e.g., "1m", "3m", "1h").
    :param int start_time: Start timestamp (milliseconds) of the time range.
    :param int end_time: End timestamp (milliseconds) of the time range.
    :param int tick_interval_ms: Kline tick interval in milliseconds.
    :param int limit: API limit for the number of klines in a single request (default: 1000).
    :param bool ignore_errors: If tru, just throw a warning on error. Recommended for redis filler.
    :return: A list of klines data within the given time range.

    Example of binpan response:

     .. code-block:: python

        [[1696786200000,
         '27890.28000000',
         '27892.47000000',
         '27890.27000000',
         '27890.45000000',
         '5.42396000',
         1696786259999,
         '151280.49524920',
         370,
         '1.99232000',
         '55567.81240580',
         '0']...]
    """
    start = int((start_time // tick_interval_ms) * tick_interval_ms)
    # end = int(-(end_time // -tick_interval_ms) * tick_interval_ms)  # trae una de mas
    end = int((end_time // tick_interval_ms) * tick_interval_ms)

    all_data = []

    for curr_start in range(start, end, limit * tick_interval_ms):

        curr_end = curr_start + (limit * tick_interval_ms)
        if ignore_errors:
            try:
                data = get_candles_by_time_stamps(symbol=symbol,
                                                  tick_interval=tick_interval,
                                                  start_time=curr_start,
                                                  end_time=curr_end)
            except Exception as e:
                market_logger.warning(f"{symbol} kline_{tick_interval} missing: {e}")
                continue
        else:
            data = get_candles_by_time_stamps(symbol=symbol,
                                              tick_interval=tick_interval,
                                              start_time=curr_start,
                                              end_time=curr_end)
        if data:
            all_data.extend(data)

    # ordena datos por primera columna. No debería haber
    all_data = sorted(all_data, key=lambda x: x[0])

    # elimina duplicados por primera columna. No debería haber
    all_data = [i for n, i in enumerate(all_data) if i not in all_data[n + 1:]]

    return all_data


def parse_candles_to_dataframe(raw_response: list,
                               symbol: str,
                               tick_interval: str,
                               columns: list = None,
                               time_cols: list = None,
                               time_zone: str or None = 'UTC') -> pd.DataFrame:
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
    :return:                    Pandas DataFrame

    """
    if not time_cols:
        time_cols = ['Open time', 'Close time']

    if not columns:
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote volume', 'Trades', 'Taker buy base volume',
                   'Taker buy quote volume', 'Ignore']

    columns = [col for col in columns if not 'timestamp' in col]

    # check if redis columns
    if raw_response:
        if type(raw_response[0]) is list:
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

    if type(time_zone) is str and time_zone != 'UTC':  # converts to time zone the time columns
        for col in time_cols:
            df.loc[:, col] = convert_ms_column_to_datetime_with_zone(df, col, time_zone=time_zone)
            # df.loc[:, col] = df[col].apply(lambda x:convert_datetime_to_string(x))
            df.loc[:, col] = convert_ms_column_to_datetime_with_zone(df, col, time_zone=time_zone, ambiguous='infer')

    else:
        for col in time_cols:
            df.loc[:, col] = df[col].apply(lambda x: convert_milliseconds_to_utc_string(x))

    # if time_index and time_zone:
    if time_zone:
        date_index = df['Open timestamp'].apply(convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
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

def basic_dataframe(data: pd.DataFrame, exceptions: list = None, actions_col='actions') -> pd.DataFrame:
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
    df = data.copy(deep=True)
    for col in df.columns:
        if col.endswith(' time'):
            continue
        elif col == "Date":  # when data is previously parsed
            continue
        df[col] = pd.to_numeric(arg=df[col], downcast='integer', errors='ignore')
    return df


def convert_dict_to_numeric(input_dict: dict, parse_datetime2int: bool) -> dict:
    """
    Converts all values in a dictionary to numeric values if possible.

    :param input_dict: A dictionary.
    :param parse_datetime2int: If True, parses datetime columns to int.
    :return: Returns a dictionary with all values converted to numeric values if possible.
    """
    output_dict = dict()

    for key, value in input_dict.items():
        if isinstance(value, (datetime, pd.Timestamp)):
            if parse_datetime2int:
                output_dict[key] = int(value.timestamp() * 1000)
            else:
                output_dict[key] = value
            continue
        try:
            float_val = float(value)
            # Downgrade a int si es posible
            if float_val.is_integer():
                output_dict[key] = int(float_val)
            else:
                output_dict[key] = float_val
        except Exception:
            # Si falla la conversión, mantener el valor original
            output_dict[key] = value
    return output_dict


##########
# Trades #
##########

def get_last_agg_trades(symbol: str, limit=1000) -> list:
    """
    Get just the last aggregated trades from API.

    GET /api/v3/aggTrades

    Get compressed, aggregate trades. Trades that fill at the time, from the same order, with the same price will have the quantity
    aggregated.

    Weight(IP): 1

    :param symbol: A binance symbol.
    :param limit: API max limit is 1000.
    :return list: Aggregated trades in a list.

    Return example:

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
    endpoint = '/api/v3/aggTrades?'
    check_weight(1, endpoint=endpoint)
    query = {'symbol': symbol, 'limit': limit}
    return get_response(url=endpoint, params=query)


def get_aggregated_trades(symbol: str, fromId: int = None, limit: int = None, decimal_mode: bool = False) -> list:
    """
    Returns aggregated trades from id to limit or last trades if id not specified.

    Limit applied in fromId mode defaults to 500. Maximum is 1000.

    GET /api/v3/aggTrades

    Get compressed, aggregate trades. Trades that fill at the time, from the same order, with the same price will have the quantity
    aggregated.

    Weight(IP): 1

    :param str symbol: A binance valid symbol.
    :param int fromId: Trade id to fetch from. If not passed, gets most recent trades.
    :param int limit: Count of trades to ask for.
    :param bool decimal_mode: Enables decimal type for return.
    :return list: Returns a list from the Binance API.

    Return example:

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

    endpoint = '/api/v3/aggTrades?'
    check_weight(1, endpoint=endpoint)
    query = {'symbol': symbol, 'limit': limit, 'fromId': fromId, 'recWindow': None}
    try:
        api_key, _ = get_encoded_secrets()
    except Exception as e:
        market_logger.error(f"Missing api_key: {e}")
        raise e
    return get_semi_signed_request(url=endpoint, decimal_mode=decimal_mode, api_key=api_key, params=query)


# noinspection PyUnresolvedReferences
def get_historical_agg_trades(symbol: str,
                              startTime: int = None,
                              endTime: int = None,
                              start_trade_id: int = None,
                              end_trade_id: int = None,
                              limit: int = 1000,
                              redis_client_trades=None) -> List[dict]:
    """
    Returns aggregated trades from id to limit or last trades if id not specified. Also is possible to get from starTime utc in
    milliseconds from epoch or until endtime milliseconds from epoch.

    Deprecated: If it is tested with more than 1 hour of trades, it gives error 1127 and if you adjust it to one hour,
    the maximum limit of 1000 is NOT applied.

    Update: API now limits to 1000 trades requests by id.

    Start time and end time not applied if trade id passed.

    Limit applied in fromId mode defaults to 500. Maximum is 1000.

    :param str symbol: A binance valid symbol.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int start_trade_id: A trade id as first one (older).
    :param int end_trade_id: A trade id as last one (newer).
    :param int limit: Count of trades to ask for. Ignored if start and end passed.
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
    try:
        assert bool(startTime or endTime) ^ bool(start_trade_id or end_trade_id)
    except AssertionError:
        raise BinPanException(f"BinPan Exception: get_historical_agg_trades params mixed time, timestamp and trade id: {locals()}")

    if start_trade_id and not end_trade_id:
        end_trade_id = start_trade_id + limit
    elif end_trade_id and not start_trade_id:
        start_trade_id = end_trade_id - limit

    requests_cnt = 1

    if start_trade_id and end_trade_id:
        trades = get_aggregated_trades(symbol=symbol, fromId=end_trade_id - 999, limit=1000)
        trade_ids = [i['a'] for i in trades]  # verificar q viene con 'id'
        if not end_trade_id in trade_ids:
            market_logger.warning(f"Trade id {end_trade_id} not reported by API. Fetching older trades.")
            end_trade_id = trade_ids[-1]
            assert end_trade_id >= start_trade_id, f"BinPan Exception: get_historical_agg_trades: end_trade_id {end_trade_id} " \
                                                   f"must be greater than start_trade_id {start_trade_id}"
        current_first_trade = trades[0]['a']
        while current_first_trade > start_trade_id:
            requests_cnt += 1
            market_logger.info(f"Requests to API for aggregate trades of {symbol}: {requests_cnt}")
            fetched_older_trades = get_aggregated_trades(symbol=symbol, fromId=(current_first_trade - 1000), limit=1000)
            trades = fetched_older_trades + trades
            current_first_trade = trades[0]['a']

        ret = [i for i in trades if start_trade_id <= i['a'] <= end_trade_id]
        return sorted(ret, key=lambda x: x['a'])

    # # trades = get_last_agg_trades(symbol=symbol, limit=1000)
    # if start_trade_id and end_trade_id:
    #     trades = get_aggregated_trades(symbol=symbol, fromId=end_trade_id-1000, limit=1000)
    # else:
    #     trades = get_last_agg_trades(symbol=symbol, limit=1000)
    #
    # # with trade ids, find start
    # if start_trade_id and end_trade_id and not redis_client_trades:
    #     current_first_trade = trades[0]['a']
    #     while current_first_trade > start_trade_id:
    #         requests_cnt += 1
    #         market_logger.info(f"Requests to API for aggregated trades of {symbol}: {requests_cnt}")
    #         fetched_older_trades = get_aggregated_trades(symbol=symbol, fromId=(current_first_trade - 1000), limit=1000)
    #         trades = fetched_older_trades + trades
    #         current_first_trade = trades[0]['a']
    #
    #     current_last_trade = trades[-1]['a']
    #     while current_last_trade < end_trade_id:
    #         requests_cnt += 1
    #         market_logger.info(f"Requests to API for aggregated trades of {symbol}: {requests_cnt}")
    #         fetched_older_trades = get_aggregated_trades(symbol=symbol, fromId=current_last_trade, limit=1000)
    #         trades = fetched_older_trades + trades
    #         current_last_trade = trades[-1]['a']
    #
    #     ret = [i for i in trades if start_trade_id <= i['a'] <= end_trade_id]
    #     return sorted(ret, key=lambda x: x['a'])

    # with timestamps or trade ids
    elif not redis_client_trades and (startTime or endTime):
        trades = get_last_agg_trades(symbol=symbol, limit=1000)
        current_first_trade_time = trades[0]['T']
        current_first_trade = trades[0]['a']
        # if startTime or endTime:
        if startTime:
            while current_first_trade_time >= startTime:
                requests_cnt += 1
                market_logger.info(f"Requests API for aggregated trades searching STARTIME {symbol}: {requests_cnt} current_first_trade:"
                                   f"{current_first_trade}")
                fetched_older_trades = get_aggregated_trades(symbol=symbol, fromId=current_first_trade - 1000, limit=1000)
                trades = fetched_older_trades + trades
                current_first_trade_time = trades[0]['T']
                current_first_trade = trades[0]['a']
        if endTime:
            current_last_trade = trades[-1]['a']
            prev_last_trade = current_last_trade
            current_last_trade_time = trades[-1]['T']
            retry_count = 0  # añadido

            while current_last_trade_time <= endTime:
                requests_cnt += 1
                market_logger.info(f"Requests API for aggregated trades searching ENDTIME {symbol}: "
                                   f"{requests_cnt}  current_last_trade:{current_last_trade}")
                fetched_newer_trades = get_aggregated_trades(symbol=symbol, fromId=current_last_trade, limit=1000)
                trades += fetched_newer_trades
                current_last_trade = trades[-1]['a']
                current_last_trade_time = trades[-1]['T']
                if current_last_trade != prev_last_trade:
                    prev_last_trade = current_last_trade  # vigilando esto
                    retry_count = 0
                else:
                    retry_count += 1
                    if retry_count >= 3:
                        break

        ret = [i for i in trades if startTime <= i['T'] <= endTime]
        return sorted(ret, key=lambda x: x['a'])

    # from redis
    response = []
    market_logger.info(f"Fetching aggregated trades from redis server for {symbol}")

    curr_trades = redis_client_trades.zrange(name=f"{symbol.lower()}@aggTrade", start=0, end=-1, withscores=True)

    curr_trades = [json.loads(i[0]) for i in curr_trades]
    market_logger.debug(f"Fetched {len(curr_trades)} aggregated trades for {symbol}")

    if curr_trades:
        timestamps_dict = {k['T']: k['a'] for k in curr_trades}  # associate timestamps with trade id
        trade_ids = [i for t, i in timestamps_dict.items() if startTime <= t <= endTime]
        market_logger.debug(f"List of aggregated trades {len(trade_ids)} for {symbol}")

        if trade_ids:
            response = redis_client_trades.zrangebyscore(name=f"{symbol.lower()}@aggTrade", min=trade_ids[0], max=trade_ids[
                -1], withscores=False)
            response = [json.loads(i) for i in response]
        else:
            response = []

        market_logger.info(f"Clean aggregated {len(response)} trades found for {symbol}")

    if not response:
        start_str = convert_milliseconds_to_utc_string(ms=startTime)
        end_str = convert_milliseconds_to_utc_string(ms=endTime)
        market_logger.info(f"No aggregated trade IDs found for {symbol} for given interval {start_str} and {end_str} (UTC) in server.")
    return response


def parse_agg_trades_to_dataframe(response: list, columns: dict, symbol: str, time_zone: str = None, time_index: bool = None,
                                  drop_dupes: str = None):
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
    :param str drop_dupes: Drop duplicated by a column name.
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
        df.loc[:, col] = convert_ms_column_to_datetime_with_zone(df, col, time_zone=time_zone)
        df.loc[:, col] = df[col].apply(lambda x: convert_datetime_to_string(x))
    else:
        df.loc[:, col] = df[timestamps_col].apply(lambda x: convert_milliseconds_to_utc_string(x))

    if time_index:
        date_index = timestamps_serie.apply(convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
        df.set_index(date_index, inplace=True)

    index_name = f"{symbol} {time_zone}"
    df.index.name = index_name
    # df.loc[:, 'Buyer was maker'] = df['Buyer was maker'].astype(bool)

    if drop_dupes:
        df.drop_duplicates(subset=drop_dupes, keep='last', inplace=True)

    # return df[['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Date', 'Timestamp', 'Buyer was maker',
    #            'Best price match']]
    return df[agg_trades_columns_from_binance]


def get_last_atomic_trades(symbol: str, limit=1000) -> list:
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
    try:
        api_key, _ = get_encoded_secrets()
    except Exception as e:
        market_logger.error(f"Missing api_key: {e}")
        raise e
    return get_semi_signed_request(url=endpoint, decimal_mode=decimal_mode, api_key=api_key, params=query)


def get_historical_atomic_trades(symbol: str,
                                 startTime: int = None,
                                 endTime: int = None,
                                 start_trade_id: int = None,
                                 end_trade_id: int = None,
                                 limit: int = 1000) -> List[dict]:
    """
    Returns atomic (not aggregated) trades between timestamps. It iterates over limit 1000 intervals to adjust to API limit.

    This request can be very slow because the API request weight limit.

    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int start_trade_id: A trade id as first one (older).
    :param int end_trade_id: A trade id as last one (newer).
    :param int limit: Limit for missing heads or tails of the interval requested with timestamps or trade ids. Ignored if  start and end
     passed.
    :param str symbol: A binance valid symbol.
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
    retry_limit = 10

    try:
        assert bool(startTime or endTime) ^ bool(start_trade_id or end_trade_id)
    except AssertionError:
        raise BinPanException(f"BinPan Exception: get_historical_atomic_trades params mixed time, timestamp and trade id: {locals()}")

    if start_trade_id and not end_trade_id:
        end_trade_id = start_trade_id + limit
    elif end_trade_id and not start_trade_id:
        start_trade_id = end_trade_id - limit

    requests_cnt = 1

    if start_trade_id and end_trade_id:
        trades = get_atomic_trades(symbol=symbol, fromId=end_trade_id - 999, limit=1000)
        trade_ids = [i['id'] for i in trades]
        if not end_trade_id in trade_ids:
            market_logger.warning(f"Trade id {end_trade_id} not reported by API. Fetching older trades.")
            end_trade_id = trade_ids[-1]
            assert end_trade_id >= start_trade_id, f"BinPan Exception: get_historical_atomic_trades: end_trade_id {end_trade_id} " \
                                                   f"must be greater than start_trade_id {start_trade_id}"
        current_first_trade = trades[0]['id']
        while current_first_trade > start_trade_id:
            requests_cnt += 1
            market_logger.info(f"Requests to API for atomic trades of {symbol}: {requests_cnt}")
            fetched_older_trades = get_atomic_trades(symbol=symbol, fromId=(current_first_trade - 1000), limit=1000)
            trades = fetched_older_trades + trades
            current_first_trade = trades[0]['id']

        ret = [i for i in trades if start_trade_id <= i['id'] <= end_trade_id]
        return sorted(ret, key=lambda x: x['id'])

    # if start_trade_id and end_trade_id:
    #     trades = get_atomic_trades(symbol=symbol, fromId=end_trade_id - 1000, limit=1000)
    # else:
    #     trades = get_last_atomic_trades(symbol=symbol, limit=1000)
    # requests_cnt = 0
    #
    # # with trade ids, find start
    # if start_trade_id and end_trade_id:
    #     current_first_trade = trades[0]['id']
    #     while current_first_trade > start_trade_id:
    #         requests_cnt += 1
    #         market_logger.info(f"Requests to API for atomic trades of {symbol}: {requests_cnt}")
    #         fetched_older_trades = get_atomic_trades(symbol=symbol, fromId=(current_first_trade - 1000), limit=1000)
    #         trades = fetched_older_trades + trades
    #         current_first_trade = trades[0]['id']
    #
    #     current_last_trade = trades[-1]['id']
    #     while current_last_trade < end_trade_id:
    #         requests_cnt += 1
    #         market_logger.info(f"Requests to API for atomic trades of {symbol}: {requests_cnt}")
    #         fetched_older_trades = get_atomic_trades(symbol=symbol, fromId=current_last_trade, limit=1000)
    #         trades = fetched_older_trades + trades
    #         current_last_trade = trades[-1]['id']
    #
    #     ret = [i for i in trades if start_trade_id <= i['id'] <= end_trade_id]
    #     return sorted(ret, key=lambda x: x['id'])

    # with timestamps or trade ids
    elif startTime or endTime:
        trades = get_last_atomic_trades(symbol=symbol, limit=1000)
        current_first_trade_time = trades[0]['time']
        current_first_trade = trades[0]['id']

        if startTime:
            while current_first_trade_time >= startTime:
                requests_cnt += 1
                market_logger.info(f"Requests API for atomic trades searching STARTIME {symbol}: {requests_cnt} current_first_trade:"
                                   f"{current_first_trade}")
                fetched_older_trades = get_atomic_trades(symbol=symbol, fromId=current_first_trade - 1000, limit=1000)
                trades = fetched_older_trades + trades
                current_first_trade_time = trades[0]['time']
                current_first_trade = trades[0]['id']
        if endTime:
            current_last_trade = trades[-1]['id']
            prev_last_trade = current_last_trade
            current_last_trade_time = trades[-1]['time']
            retry_count = 0

            while current_last_trade_time <= endTime:
                requests_cnt += 1
                market_logger.info(f"Requests API for atomic trades searching ENDTIME {symbol}: {requests_cnt} current_last_trade:"
                                   f"{current_last_trade}")
                fetched_newer_trades = get_atomic_trades(symbol=symbol, fromId=current_last_trade, limit=1000)
                trades += fetched_newer_trades
                current_last_trade = trades[-1]['id']
                current_last_trade_time = trades[-1]['time']
                if current_last_trade != prev_last_trade:
                    prev_last_trade = current_last_trade  # vigilando esto
                    retry_count = 0
                else:
                    retry_count += 1
                    if retry_count >= 3:
                        break

        ret = [i for i in trades if startTime <= i['time'] <= endTime]
        return sorted(ret, key=lambda x: x['id'])

    # from redis
    # response = []
    # market_logger.info(f"Fetching atomic trades from redis server for {symbol}")
    #
    # curr_trades = redis_client_trades.zrange(name=f"{symbol.lower()}@trade", start=0, end=-1, withscores=True)
    #
    # curr_trades = [json.loads(i[0]) for i in curr_trades]
    # market_logger.debug(f"Fetched {len(curr_trades)} atomic trades for {symbol}")
    #
    # if curr_trades:
    #     timestamps_dict = {k['T']: k['t'] for k in curr_trades}  # associate timestamps with trade id
    #     trade_ids = [i for t, i in timestamps_dict.items() if startTime <= t <= endTime]
    #     market_logger.debug(f"List of atomic trades {len(trade_ids)} for {symbol}")
    #
    #     if trade_ids:
    #         response = redis_client_trades.zrangebyscore(name=f"{symbol.lower()}@trade", min=trade_ids[0], max=trade_ids[
    #             -1], withscores=False)
    #         response = [json.loads(i) for i in response]
    #     else:
    #         response = []
    #
    #     market_logger.info(f"Clean atomic {len(response)} trades found for {symbol}")
    #
    # if not response:
    #     start_str = convert_milliseconds_to_utc_string(ms=startTime)
    #     end_str = convert_milliseconds_to_utc_string(ms=endTime)
    #     market_logger.info(f"No atomic trade IDs found for {symbol} for given interval {start_str} and {end_str} (UTC) in server.")
    # return response


def parse_atomic_trades_to_dataframe(response: list,
                                     columns: dict,
                                     symbol: str,
                                     time_zone: str = None,
                                     time_index: bool = None,
                                     drop_dupes: str = None):
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
    :param str drop_dupes: Drop duplicated by a column name.
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
        df.loc[:, col] = convert_ms_column_to_datetime_with_zone(df, col, time_zone=time_zone)
        df.loc[:, col] = df[col].apply(lambda x: convert_datetime_to_string(x))
    else:
        df.loc[:, col] = df[timestamps_col].apply(lambda x: convert_milliseconds_to_utc_string(x))

    if time_index:
        date_index = timestamps_serie.apply(convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
        df.set_index(date_index, inplace=True)

    index_name = f"{symbol} {time_zone}"
    df.index.name = index_name

    if drop_dupes:
        df.drop_duplicates(subset=drop_dupes, keep='last', inplace=True)

    if 'quoteQty' in columns.keys():
        # return df[['Trade Id', 'Price', 'Quantity', 'Quote quantity', 'Date', 'Timestamp', 'Buyer was maker', 'Best price match']]
        return df[atomic_trades_columns_from_binance]
    else:  # it was a redis response
        # return df[['Trade Id', 'Price', 'Quantity', 'Buyer Order Id', 'Seller Order Id', 'Date', 'Timestamp', 'Buyer was maker',
        #            'Best price match']]
        return df[atomic_trades_columns_from_redis]


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
    if type(response) is dict:
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


def intermediate_conversion(coin: str, decimal_mode: bool, prices: dict = None, try_coin: str = 'BTC',
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


def convert_coin(coin: str, decimal_mode: bool, convert_to: str = 'BUSD', coin_qty: float or dd = 1, prices: dict = None) -> float or None:
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
