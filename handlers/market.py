"""

Market functions

"""

from time import time
import pandas as pd
from random import randint
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
                               time_zone='Europe/Madrid') -> List[list]:
    """
    Calls API for a candles list using one or two timestamps, starting and ending.

    In case the limit is passed and exceeded by requested time intervals, the start_time prevails over the end_time,
    start_time must come in milliseconds from epoch.

    In case of two timeStamps as arguments, limit is ignored and a loop is performed to get all candles in the range.

    The API rounds the startTime up to the next open of the next candle. That is, it does not include the candle in which there is
    that timeStamp, but the next candle of the corresponding tick_interval, except in case it exactly matches the value of an open
    timestamp, in which case it will include it in the return.

    The indicated endTime will include the candlestick that timestamp is on. It should be in milliseconds. It can be a not closed one if
    is open right in between the endtime timestamp. API is inclusive in this case.

    If no timestamps are passed, the last quantity candlesticks up to limit count are returned.

    :param str symbol: A binance valid symbol.
    :param str tick_interval: A binance valid time interval for candlesticks.
    :param int start_time: A timestamp in milliseconds from epoch.
    :param int end_time: A timestamp in milliseconds from epoch.
    :param int limit: Count of candles to ask for.
    :param str time_zone: Just used for exception errors.
    :return list: Returns a list from the Binance API.

    .. note::    By default unique API response is inclusive for start_time and end_time, IF LESS than 1000 limit applied.

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
    tick_milliseconds = int(tick_seconds[tick_interval] * 1000)
    limit = min(limit, 1000)

    if not start_time and not end_time:
        end_time = int(time()) * 1000  # la api traería la ultima vela cerrada
        start_time = end_time - ((limit - 1) * tick_milliseconds)
        start_time = open_from_milliseconds(ms=start_time, tick_interval=tick_interval)

    if not start_time and end_time:
        start_time = end_time - ((limit - 1) * tick_milliseconds)
        start_time = open_from_milliseconds(ms=start_time, tick_interval=tick_interval)

    elif start_time and not end_time:
        end_time = start_time + ((limit - 1) * tick_milliseconds)  # end time is included
        end_time = open_from_milliseconds(ms=end_time, tick_interval=tick_interval)

    end_time = min(end_time,
                   next_open_by_milliseconds(ms=int(time() * 1000), tick_interval=tick_interval),
                   next_open_by_milliseconds(ms=end_time, tick_interval=tick_interval))

    start_string = convert_milliseconds_to_str(ms=start_time, timezoned=time_zone)
    end_string = convert_milliseconds_to_str(ms=end_time, timezoned=time_zone)

    # chan = f"{symbol.lower()}@kline_{tick_interval}"
    market_logger.info(f"get_candles_by_time_stamps -> symbol={symbol} tick_interval={tick_interval} start={start_string} end="
                       f"{end_string} limit={limit}")

    # prepare iteration for big loops
    ranges = [(i, i + (1000 * tick_milliseconds)) for i in range(start_time, end_time, tick_milliseconds * 1000)]

    # loop
    raw_candles = []
    for start, end in ranges:

        params = {'symbol': symbol, 'interval': tick_interval, 'startTime': start, 'endTime': end, 'limit': limit}
        params = {k: v for k, v in params.items() if v}
        check_weight(1, endpoint=endpoint)

        start_str = convert_milliseconds_to_str(start, timezoned=time_zone)
        end_str = convert_milliseconds_to_str(min(end, int(1000 * time())), timezoned=time_zone)

        # expected_klines = int((end - start) / tick_milliseconds)
        expected_klines = int(-((end - start) // -tick_milliseconds))
        market_logger.debug(f"API request: {symbol} {start_str} to {end_str}. Expected klines: {expected_klines}")

        response = get_response(url=endpoint, params=params)

        if len(response) < expected_klines:
            market_logger.warning(f"API response missing {expected_klines - len(response)} klines for {symbol} {start_str} to "
                                  f"{end_str} expected {expected_klines} got {len(response)}")
        raw_candles += response

    if not raw_candles:
        msg = (f"BinPan: Missing in API requested klines for {symbol.lower()}@kline_{tick_interval} between {start_string} "
               f"and {end_string}")
        market_logger.warning(msg)
        return []
        # raise Exception(msg)

    # descarta sobrantes
    overtime_candle_ts = next_open_by_milliseconds(ms=end_time, tick_interval=tick_interval)

    if type(raw_candles[0]) is list:  # if from binance
        raw_candles_ = [i for i in raw_candles if int(i[0]) < overtime_candle_ts]
    else:
        open_ts_key = list(raw_candles[0].keys())[0]
        raw_candles_ = [i for i in raw_candles if int(i[open_ts_key]) < overtime_candle_ts]

    if len(raw_candles) != len(raw_candles_):
        market_logger.info(f"Pruned overtime_candle_ts {len(raw_candles) - len(raw_candles_)} candles from API response for {symbol} "
                           f"{tick_interval}")

    return raw_candles_


def get_historical_candles(symbol: str,
                           tick_interval: str,
                           start_time: int,
                           end_time: int,
                           tick_interval_ms: int,
                           requests_limit: int = 1000,
                           ignore_errors: bool = False) -> List[list]:
    """
    Retrieve all kline data within the given time range considering the API limit.

    Start time and end time are rounded down to the nearest open tick interval and are included both opens rounded.

    Please, check last close timestamp in klines because it can be not closed yet.

    :param str symbol: The trading pair symbol (e.g., "BTCUSDT").
    :param str tick_interval: Kline tick interval (e.g., "1m", "3m", "1h").
    :param int start_time: Start timestamp (milliseconds) of the time range.
    :param int end_time: End timestamp (milliseconds) of the time range.
    :param int tick_interval_ms: Kline tick interval in milliseconds.
    :param int requests_limit: API limit for the number of klines in a single request (default: 1000).
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

    for curr_start in range(start, end, requests_limit * tick_interval_ms):

        curr_end = curr_start + (requests_limit * tick_interval_ms)
        if ignore_errors:
            try:
                # get_candles_by_time_stamps excludes end_time
                data = get_candles_by_time_stamps(symbol=symbol,
                                                  tick_interval=tick_interval,
                                                  start_time=curr_start,
                                                  end_time=curr_end)
            except Exception as e:
                market_logger.warning(f"{symbol} kline_{tick_interval} missing: {e}")
                continue
        else:
            # get_candles_by_time_stamps excludes end_time
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

def get_last_agg_trades(symbol: str, limit=1000) -> List[dict]:
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


def get_aggregated_trades(symbol: str, fromId: int = None, limit: int = None, decimal_mode: bool = False) -> List[dict]:
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


def get_historical_agg_trades(symbol: str,
                              startTime: int = None,
                              endTime: int = None,
                              start_trade_id: int = None,
                              end_trade_id: int = None,
                              limit_ids: int = 1000,
                              limit_hours: float = 1) -> List[dict]:
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
    :param int limit_ids: Count of trades to ask for if just start_trade_id or just end_trade_id passed.
        Ignored if startTime or endTime passed.
    :param int limit_hours: Count of hours to ask for if just startTime or just endTime passed.
        Ignored if start_trade_id or end_trade_id passed.
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

    id_field = "a"
    timestamp_field = "T"
    trade_type = "aggregated"

    return get_historical_trades(symbol=symbol,
                                 trade_type=trade_type,
                                 id_field=id_field,
                                 timestamp_field=timestamp_field,
                                 startTime=startTime,
                                 endTime=endTime,
                                 start_trade_id=start_trade_id,
                                 end_trade_id=end_trade_id,
                                 limit_ids=limit_ids,
                                 limit_hours=limit_hours)


def get_historical_trades(symbol: str,
                          trade_type: str,
                          id_field: str,
                          timestamp_field: str,
                          startTime: int = None,
                          endTime: int = None,
                          start_trade_id: int = None,
                          end_trade_id: int = None,
                          limit_ids: int = 1000,
                          limit_hours: float = 1) -> List[dict]:
    """
    Returns aggregated or atomic trades from id to limit or last trades if id not specified. Also is possible to get from starTime utc in
    milliseconds from epoch or until endtime milliseconds from epoch.

    Start time and end time not applied if trade id passed.

    Limit applied in fromId mode defaults to 1000.

    Limit hours applied in startTime or endTime mode defaults to 1 hour.

    :param str symbol: A binance valid symbol.
    :param str trade_type: 'aggregated' or 'atomic'.
    :param str id_field: 'a' or 't' for aggregated or atomic respectively.
    :param str timestamp_field: 'T' or 't' for aggregated or atomic respectively.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int start_trade_id: A trade id as first one (older).
    :param int end_trade_id: A trade id as last one (newer).
    :param int limit_ids: Count of trades to ask for if just start_trade_id or just end_trade_id passed.
        Ignored if startTime or endTime passed.
    :param int limit_hours: Count of hours to ask for if just startTime or just endTime passed.
        Ignored if start_trade_id or end_trade_id passed.
    :return list: Returns a list from the Binance API in dicts.

    """

    requests_cnt = 1

    if trade_type == "aggregated":
        my_func = get_aggregated_trades
    else:  # atomic
        my_func = get_atomic_trades

    try:
        assert bool(startTime or endTime) ^ bool(start_trade_id or end_trade_id)
    except AssertionError:
        if not startTime and not endTime and not start_trade_id and not end_trade_id:
            raise BinPanException(f"BinPan Exception: get historical {trade_type} trades params missing: {locals()}")
        else:
            raise BinPanException(f"BinPan Exception: get historical {trade_type} trades params mixed time, timestamp and trade id: {locals()}")

    if start_trade_id and not end_trade_id:
        assert limit_ids, f"BinPan Exception: get historical {trade_type} from start_trade_id without trades limit_ids: {locals()}"
        end_trade_id = start_trade_id + (limit_ids - 1)
    elif end_trade_id and not start_trade_id:
        assert limit_ids, f"BinPan Exception: get historical {trade_type} from end_trade_id without trades limit_ids: {locals()}"
        start_trade_id = end_trade_id - (limit_ids - 1)

    if start_trade_id and end_trade_id:
        assert start_trade_id <= end_trade_id, f"BinPan Exception: get historical {trade_type} trades start_trade_id > end_trade_id: {locals()}"
        market_logger.info(f"Ignoring limit_ids {limit_ids} for {trade_type} trades of {symbol} from {start_trade_id} to {end_trade_id}")
        trades = my_func(symbol=symbol, fromId=end_trade_id - 999, limit=1000)  # el presente
        end_trade_id = trades[-1][id_field]  # evita pedir del futuro
        current_first_trade = trades[0][id_field]

        while current_first_trade > start_trade_id:
            requests_cnt += 1
            market_logger.info(f"Requests to API for {trade_type} trades of {symbol}: {requests_cnt} current_first_trade: "
                               f"{current_first_trade}")
            # restamos 1000 pq current_first_trade ya lo tenemos, asi no viene repetido
            fetched_older_trades = my_func(symbol=symbol, fromId=(current_first_trade - 1000), limit=1000)
            trades = fetched_older_trades + trades
            current_first_trade = trades[0][id_field]

        ret = [i for i in trades if start_trade_id <= i[id_field] <= end_trade_id]
        return sorted(ret, key=lambda x: x[id_field])

    # with timestamps
    elif startTime or endTime:
        factor = 1
        trades = my_func(symbol=symbol, limit=1000)
        current_first_trade_time = trades[0][timestamp_field]
        current_last_trade_time = trades[-1][timestamp_field]
        current_first_trade = trades[0][id_field]

        if endTime and not startTime:
            assert limit_hours, f"BinPan Exception: get historical {trade_type} trades endTime without trades limit_hours: {locals()}"
            market_logger.info(f"Star time not passed for {trade_type} trades of {symbol}. Calculating {limit_hours} hour ago.")
            limit_hours_ms = int(60 * 1000 * 60 * limit_hours)
            startTime = min(endTime, current_last_trade_time) - limit_hours_ms
            endTime = min(endTime, current_last_trade_time)  # no podemos pedir trades del futuro

        if startTime and not endTime:
            assert limit_hours, f"BinPan Exception: get historical {trade_type} trades startTime without trades limit_hours: {locals()}"
            market_logger.info(f"End time not passed for {trade_type} trades of {symbol}. Calculating {limit_hours} hours ahead.")
            endTime = startTime + (1000 * 60 * 60 * limit_hours)

        elif not startTime and not endTime:
            endTime = current_last_trade_time  # no podemos pedir trades del futuro
            market_logger.info(f"Start time not passed for {trade_type} trades of {symbol}. Calculating {limit_hours} hours ago.")
            startTime = endTime - (1000 * 60 * 60 * limit_hours)

        assert startTime <= endTime, f"BinPan Exception: get historical {trade_type} trades endTime < startTime: {locals()}"

        touche = False

        while startTime <= current_first_trade_time:

            requests_cnt += 1
            start_str = pd.to_datetime(current_first_trade_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            market_logger.info(f"Requests API for {trade_type} trades searching STARTIME {symbol}: {requests_cnt} current_first_trade:"
                               f"{current_first_trade} date: {start_str}")

            # restamos 1000 pq el current first trade ya lo tenemos y no queremos que venga repetido
            fetched_older_trades = my_func(symbol=symbol, fromId=current_first_trade - (1000 * factor), limit=1000)

            current_first_trade_factor = fetched_older_trades[0][id_field]
            current_last_trade_factor = fetched_older_trades[-1][id_field]
            current_first_timestamp_factor = fetched_older_trades[0][timestamp_field]
            current_last_timestamp_factor = fetched_older_trades[-1][timestamp_field]

            if endTime < current_first_timestamp_factor:
                market_logger.debug("endTime < current_first_timestamp_factor")
                if not touche:
                    factor += 1
                current_first_trade = current_first_trade_factor
                current_first_trade_time = current_first_timestamp_factor
                continue

            elif current_last_timestamp_factor < endTime and not touche:  # si nos pasamos frenamos y volvemos a subir
                market_logger.debug("current_last_timestamp_factor < endTime and not touche")
                factor = max(factor - 2, 1)
                unlock_posible_loops = randint(1, 1000)
                current_first_trade = current_last_trade_factor + unlock_posible_loops + 1000 * factor  # hay que subir, nos hemos pasado
                current_first_trade_time = max(current_first_timestamp_factor, startTime)  # para que no bloquee por saltos grandes
                continue

            elif current_first_timestamp_factor <= endTime <= current_last_timestamp_factor:  # empezamos a grabar
                market_logger.debug("current_first_timestamp_factor <= endTime <= current_last_timestamp_factor")
                factor = 1
                touche = True
                current_first_trade = current_first_trade_factor
                current_first_trade_time = current_first_timestamp_factor
                trades = fetched_older_trades + trades
                continue

            elif touche:
                market_logger.debug("touche")
                current_first_trade = current_first_trade_factor
                current_first_trade_time = current_first_timestamp_factor
                trades = fetched_older_trades + trades

            else:
                start_str = pd.to_datetime(current_first_timestamp_factor, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                end_str = pd.to_datetime(current_last_timestamp_factor, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                raise BinPanException(f"BinPan Exception: {symbol} get historical {trade_type} trades {start_str} - {end_str}")

        # acota
        ret = [i for i in trades if startTime <= i[timestamp_field] <= endTime]
        # elimina trades duplicados
        ret = [i for n, i in enumerate(ret) if i not in ret[n + 1:]]
        # ordena
        return sorted(ret, key=lambda x: x[id_field])


def parse_agg_trades_to_dataframe(response: list, columns: dict, symbol: str, time_zone: str = None, time_index: bool = None,
                                  drop_dupes: str = None) -> pd.DataFrame:
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


def get_last_atomic_trades(symbol: str, limit=1000) -> List[dict]:
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
                      decimal_mode: bool = False) -> List[dict]:
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
                                 limit_ids: int = 1000,
                                 limit_hours: float = 1) -> List[dict]:
    """
    Returns atomic (not aggregated) trades between timestamps. It iterates over limit 1000 intervals to adjust to API limit.

    This request can be very slow because the API request weight limit.

    :param str symbol: A binance valid symbol.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int start_trade_id: A trade id as first one (older).
    :param int end_trade_id: A trade id as last one (newer).
    :param int limit_ids: Count of trades to ask for if just start_trade_id or just end_trade_id passed.
        Ignored if startTime or endTime passed.
    :param int limit_hours: Count of hours to ask for if just startTime or just endTime passed.
        Ignored if start_trade_id or end_trade_id passed.
    :return list: Returns a list from the Binance API in dicts.
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

    id_field = "id"
    timestamp_field = "time"
    trade_type = "atomic"

    return get_historical_trades(symbol=symbol,
                                 trade_type=trade_type,
                                 id_field=id_field,
                                 timestamp_field=timestamp_field,
                                 startTime=startTime,
                                 endTime=endTime,
                                 start_trade_id=start_trade_id,
                                 end_trade_id=end_trade_id,
                                 limit_ids=limit_ids,
                                 limit_hours=limit_hours)


def parse_atomic_trades_to_dataframe(response: list,
                                     columns: dict,
                                     symbol: str,
                                     time_zone: str = None,
                                     time_index: bool = None,
                                     drop_dupes: str = None) -> pd.DataFrame:
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


if __name__ == '__main__':
    symbol = "renusdt".upper()

    tests = [
        (30140069, None, 1000),
        (None, 30140069, None),
        (30140069 - 2000, 30140069, None),
        (30140069 - 2000, 30140069, 1000),
        (None, None, 1000),
        (None, None, None),
        (None, None, 1),]



    for st, ed, l in tests:
        print(f"Start: {st} End: {ed} Limir: {l}")
        # a = get_historical_atomic_trades("renusdt".upper(), startTime = 1699381564859- 1000, endTime=1699381564859)
        a = get_historical_atomic_trades(symbol=symbol, start_trade_id=st, end_trade_id=ed, limit_ids=l)
        print("Hours:", (a[-1]['time'] - a[0]['time']) / (1000 * 60 * 60))
        print(len(a))
        assert len(pd.Series([i["id"] for i in a]).diff().value_counts()) == 1