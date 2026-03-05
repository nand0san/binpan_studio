"""

Market functions

"""

from time import time
import pandas as pd
from decimal import Decimal as dd
from datetime import datetime

from panzer import BinancePublicClient

from .exceptions import BinPanException
from .logs import LogManager
from .quest import get_semi_signed_request
from .time_helper import (tick_seconds, convert_milliseconds_to_str, convert_ms_column_to_datetime_with_zone,
                          convert_milliseconds_to_utc_string, convert_datetime_to_string, open_from_milliseconds, next_open_by_milliseconds,
                          convert_milliseconds_to_time_zone_datetime)
from .standards import (klines_api_map_columns, agg_trades_columns_from_binance, atomic_trades_columns_from_binance,
                        atomic_trades_columns_from_redis)
from .files import get_encoded_secrets

market_logger = LogManager(filename='./logs/market_logger.log', name='market_logger', info_level='INFO')

# cliente panzer compartido (lazy singleton)
_panzer_client = None


def _get_panzer(market: str = "spot") -> BinancePublicClient:
    """Returns the shared panzer client, creating it on first use."""
    global _panzer_client
    if _panzer_client is None:
        _panzer_client = BinancePublicClient(market=market)
    return _panzer_client


##########
# Prices #
##########


def get_last_price(symbol: str = None) -> dict | list:
    """
    Returns all prices of symbols in a dict or if symbol specified, the float price of symbol.

    :param str symbol: A binance symbol.
    :return dict or list:

    """
    client = _get_panzer()
    params = {'symbol': symbol.upper()} if symbol else {}
    res = client.get('/api/v3/ticker/price', params=params, weight=2 if not symbol else 1)
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
    client = _get_panzer()
    ret = client.get('/api/v3/ticker/price', weight=2)
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
                               time_zone='Europe/Madrid') -> list[list]:
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
    # limit = min(limit, 1000)
    limit = (end_time - start_time) // tick_milliseconds

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

    market_logger.info(f"get_candles_by_time_stamps -> symbol={symbol} tick_interval={tick_interval} start={start_string} end="
                       f"{end_string} limit={limit}")

    client = _get_panzer()

    # preparar iteración por bloques de 1000 velas
    ranges = [(i, i + (1000 * tick_milliseconds)) for i in range(start_time, end_time, tick_milliseconds * 1000)]

    raw_candles = []
    for start, end in ranges:
        loop_limit = min(limit, 1000)
        end = min(end, int(1000 * time()), end_time)

        start_str = convert_milliseconds_to_str(start, timezoned=time_zone)
        end_str = convert_milliseconds_to_str(end, timezoned=time_zone)

        expected_klines = min(int(-((end - start) // -tick_milliseconds)), loop_limit)
        market_logger.debug(f"API request: {symbol} {start_str} to {end_str}. Expected klines: {expected_klines}")

        response = client.klines(symbol=symbol, interval=tick_interval, start_time=start, end_time=end, limit=loop_limit)

        if len(response) < expected_klines:
            market_logger.warning(f"API response missing {expected_klines - len(response)} klines for {symbol} {start_str} to "
                                  f"{end_str} expected {expected_klines} got {len(response)}")
        raw_candles += response

    if not raw_candles:
        msg = (f"BinPan: Missing in API requested klines for {symbol.lower()}@kline_{tick_interval} between {start_string} "
               f"and {end_string}")
        market_logger.warning(msg)
        return []

    # descarta sobrantes
    overtime_candle_ts = next_open_by_milliseconds(ms=end_time, tick_interval=tick_interval)

    if type(raw_candles[0]) is list:
        raw_candles_ = [i for i in raw_candles if int(i[0]) < overtime_candle_ts]
    else:
        open_ts_key = list(raw_candles[0].keys())[0]
        raw_candles_ = [i for i in raw_candles if int(i[open_ts_key]) < overtime_candle_ts]

    if len(raw_candles) != len(raw_candles_):
        market_logger.info(f"Pruned not closed or overtime candles {len(raw_candles) - len(raw_candles_)} candles from API response for {symbol} "
                           f"{tick_interval}")

    return raw_candles_


def get_historical_candles(symbol: str,
                           tick_interval: str,
                           start_time: int,
                           end_time: int,
                           tick_interval_ms: int,
                           requests_limit: int = 1000,
                           ignore_errors: bool = False) -> list[list]:
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
                               time_zone: str | None = 'UTC') -> pd.DataFrame:
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
                columns = list(klines_api_map_columns.keys())
                df = pd.DataFrame(raw_response, columns=columns)
                df.rename(columns=klines_api_map_columns, inplace=True)
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
            df[col] = convert_ms_column_to_datetime_with_zone(df, col, time_zone=time_zone, ambiguous='infer')

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
        try:
            df[col] = pd.to_numeric(arg=df[col])
        except (ValueError, TypeError):
            pass
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

def get_last_agg_trades(symbol: str, limit=1000) -> list[dict]:
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
    client = _get_panzer()
    return client.agg_trades(symbol=symbol, limit=limit)


def get_aggregated_trades(symbol: str, fromId: int = None, limit: int = None, decimal_mode: bool = False) -> list[dict]:
    """
    Returns aggregated trades from id to limit or last trades if id not specified.

    Limit applied in fromId mode defaults to 500. Maximum is 1000.

    GET /api/v3/aggTrades

    Get compressed, aggregate trades. Trades that fill at the time, from the same order, with the same price will have the quantity
    aggregated.

    Weight(IP): 4

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
    query = {'symbol': symbol, 'limit': limit, 'fromId': fromId, 'recWindow': None}
    try:
        api_key, _ = get_encoded_secrets()
    except Exception as e:
        market_logger.error(f"Missing api_key: {e}")
        raise e
    return get_semi_signed_request(url=endpoint, decimal_mode=decimal_mode, api_key=api_key, params=query)


def get_aggregated_trades_by_time(symbol: str, startTime: int, endTime: int, limit: int = 1000) -> list[dict]:
    """
    Returns aggregated trades using native startTime/endTime parameters of the API.

    The API limits each request to a maximum window of 1 hour and 1000 results.

    GET /api/v3/aggTrades

    Weight(IP): 4

    :param str symbol: A binance valid symbol.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int limit: Max trades per request (API max 1000).
    :return list[dict]: Returns a list of aggregated trade dicts from the Binance API.
    """
    client = _get_panzer()
    return client.agg_trades(symbol=symbol, start_time=startTime, end_time=endTime, limit=limit)


def get_historical_agg_trades(symbol: str,
                              startTime: int = None,
                              endTime: int = None,
                              start_trade_id: int = None,
                              end_trade_id: int = None,
                              limit_ids: int = 1000,
                              limit_hours: float = 1) -> list[dict]:
    """
    Returns aggregated trades between timestamps or trade IDs.

    For timestamp mode, uses native startTime/endTime API parameters with 1-hour chunking.
    For ID mode, paginates forward with fromId.

    :param str symbol: A binance valid symbol.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int start_trade_id: A trade id as first one (older).
    :param int end_trade_id: A trade id as last one (newer).
    :param int limit_ids: Count of trades to ask for if just start_trade_id or just end_trade_id passed.
        Ignored if startTime or endTime passed.
    :param float limit_hours: Count of hours to ask for if just startTime or just endTime passed.
        Ignored if start_trade_id or end_trade_id passed.
    :return list[dict]: Returns a list from the Binance API in dicts.

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

    ONE_HOUR_MS = 3_600_000
    id_field = "a"

    has_time = bool(startTime or endTime)
    has_ids = bool(start_trade_id or end_trade_id)

    if not has_time and not has_ids:
        raise BinPanException("BinPan Exception: get_historical_agg_trades sin parámetros de tiempo ni de ID")
    if has_time and has_ids:
        raise BinPanException("BinPan Exception: get_historical_agg_trades mezcla parámetros de tiempo e ID")

    ################
    # Modo por IDs #
    ################

    if has_ids:
        if start_trade_id and not end_trade_id:
            end_trade_id = start_trade_id + (limit_ids - 1)
        elif end_trade_id and not start_trade_id:
            start_trade_id = end_trade_id - (limit_ids - 1)

        assert start_trade_id <= end_trade_id, (
            f"BinPan Exception: aggTrades start_trade_id ({start_trade_id}) > end_trade_id ({end_trade_id})")

        market_logger.info(f"Obteniendo aggTrades de {symbol} por ID: {start_trade_id} -> {end_trade_id}")

        all_trades = []
        current_id = start_trade_id
        requests_cnt = 0

        while current_id <= end_trade_id:
            batch = get_aggregated_trades(symbol=symbol, fromId=current_id, limit=1000)
            requests_cnt += 1
            if not batch:
                break
            all_trades.extend(batch)
            current_id = batch[-1][id_field] + 1
            market_logger.info(f"Peticiones API aggTrades {symbol}: {requests_cnt}")
            if len(batch) < 1000:
                break

        ret = [t for t in all_trades if start_trade_id <= t[id_field] <= end_trade_id]
        return sorted(ret, key=lambda x: x[id_field])

    #######################
    # Modo por timestamps #
    #######################

    now_ms = int(time() * 1000)
    limit_hours_ms = int(ONE_HOUR_MS * limit_hours)

    if endTime and not startTime:
        endTime = min(endTime, now_ms)
        startTime = endTime - limit_hours_ms
    elif startTime and not endTime:
        endTime = min(startTime + limit_hours_ms, now_ms)

    assert startTime <= endTime, (
        f"BinPan Exception: aggTrades startTime ({startTime}) > endTime ({endTime})")

    market_logger.info(
        f"Obteniendo aggTrades de {symbol} por tiempo: "
        f"{convert_milliseconds_to_utc_string(startTime)} -> {convert_milliseconds_to_utc_string(endTime)}")

    all_trades = []
    chunk_start = startTime
    requests_cnt = 0

    while chunk_start < endTime:
        chunk_end = min(chunk_start + ONE_HOUR_MS, endTime)
        batch = get_aggregated_trades_by_time(symbol, chunk_start, chunk_end)
        requests_cnt += 1

        if not batch:
            chunk_start = chunk_end + 1
            continue

        all_trades.extend(batch)

        # sub-paginar dentro de la hora si hay >=1000 trades
        if len(batch) == 1000:
            while True:
                sub_start = batch[-1]['T'] + 1
                if sub_start > chunk_end:
                    break
                batch = get_aggregated_trades_by_time(symbol, sub_start, chunk_end)
                requests_cnt += 1
                if not batch:
                    break
                all_trades.extend(batch)
                if len(batch) < 1000:
                    break

        chunk_start = chunk_end + 1

    market_logger.info(f"aggTrades {symbol}: {requests_cnt} peticiones, {len(all_trades)} trades obtenidos")

    # deduplicar O(n) por campo 'a'
    seen = set()
    result = [t for t in all_trades if t[id_field] not in seen and not seen.add(t[id_field])]
    return sorted(result, key=lambda x: x[id_field])


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
    df[col] = df[timestamps_col]
    if time_zone != 'UTC':  # converts to time zone the time columns
        df[col] = convert_ms_column_to_datetime_with_zone(df, col, time_zone=time_zone)
        df[col] = df[col].apply(lambda x: convert_datetime_to_string(x))
    else:
        df[col] = df[timestamps_col].apply(lambda x: convert_milliseconds_to_utc_string(x))

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


def get_last_atomic_trades(symbol: str, limit=1000) -> list[dict]:
    """
    Returns recent atomic (not aggregated) trades.

    GET /api/v3/trades

    Get recent trades.

    Weight(IP): 25

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

    client = _get_panzer()
    return client.trades(symbol=symbol, limit=limit)


def get_atomic_trades(symbol: str,
                      fromId: int = None,
                      limit: int = None,
                      decimal_mode: bool = False) -> list[dict]:
    """
    Returns atomic (not aggregated) trades from id to limit or last trades if id not specified.

    Limit applied in fromId mode defaults to 500. Maximum is 1000.

    GET /api/v3/historicalTrades

    Get older market trades.

    Weight(IP): 25

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
                                 limit_hours: float = 1) -> list[dict]:
    """
    Returns atomic (not aggregated) trades between timestamps or trade IDs.

    For timestamp mode, uses aggTrades (weight 4) to discover the atomic trade ID range,
    then paginates with /api/v3/historicalTrades (weight 25) using fromId.
    This avoids the old adaptive search algorithm and is deterministic.

    For ID mode, paginates forward with fromId directly.

    :param str symbol: A binance valid symbol.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param int start_trade_id: A trade id as first one (older).
    :param int end_trade_id: A trade id as last one (newer).
    :param int limit_ids: Count of trades to ask for if just start_trade_id or just end_trade_id passed.
        Ignored if startTime or endTime passed.
    :param float limit_hours: Count of hours to ask for if just startTime or just endTime passed.
        Ignored if start_trade_id or end_trade_id passed.
    :return list[dict]: Returns a list from the Binance API in dicts.

    .. code-block::

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

    ONE_HOUR_MS = 3_600_000
    id_field = "id"
    timestamp_field = "time"

    has_time = bool(startTime or endTime)
    has_ids = bool(start_trade_id or end_trade_id)

    if not has_time and not has_ids:
        raise BinPanException("BinPan Exception: get_historical_atomic_trades sin parámetros de tiempo ni de ID")
    if has_time and has_ids:
        raise BinPanException("BinPan Exception: get_historical_atomic_trades mezcla parámetros de tiempo e ID")

    ################
    # Modo por IDs #
    ################

    if has_ids:
        if start_trade_id and not end_trade_id:
            end_trade_id = start_trade_id + (limit_ids - 1)
        elif end_trade_id and not start_trade_id:
            start_trade_id = end_trade_id - (limit_ids - 1)

        assert start_trade_id <= end_trade_id, (
            f"BinPan Exception: atomic trades start_trade_id ({start_trade_id}) > end_trade_id ({end_trade_id})")

        market_logger.info(f"Obteniendo atomic trades de {symbol} por ID: {start_trade_id} -> {end_trade_id}")

        all_trades = []
        current_id = start_trade_id
        requests_cnt = 0

        while current_id <= end_trade_id:
            batch = get_atomic_trades(symbol=symbol, fromId=current_id, limit=1000)
            requests_cnt += 1
            if not batch:
                break
            all_trades.extend(batch)
            current_id = batch[-1][id_field] + 1
            market_logger.info(f"Peticiones API atomic trades {symbol}: {requests_cnt}")
            if len(batch) < 1000:
                break

        ret = [t for t in all_trades if start_trade_id <= t[id_field] <= end_trade_id]
        return sorted(ret, key=lambda x: x[id_field])

    #######################
    # Modo por timestamps #
    #######################

    now_ms = int(time() * 1000)
    limit_hours_ms = int(ONE_HOUR_MS * limit_hours)

    if endTime and not startTime:
        endTime = min(endTime, now_ms)
        startTime = endTime - limit_hours_ms
    elif startTime and not endTime:
        endTime = min(startTime + limit_hours_ms, now_ms)

    assert startTime <= endTime, (
        f"BinPan Exception: atomic trades startTime ({startTime}) > endTime ({endTime})")

    market_logger.info(
        f"Obteniendo atomic trades de {symbol} por tiempo: "
        f"{convert_milliseconds_to_utc_string(startTime)} -> {convert_milliseconds_to_utc_string(endTime)}")

    # descubrir rango de IDs atómicos usando aggTrades (peso 4, barato)
    # primer aggTrade en startTime
    search_end = min(startTime + ONE_HOUR_MS, endTime)
    first_agg = get_aggregated_trades_by_time(symbol, startTime, search_end, limit=1)

    # si no hay trades en la primera hora, ampliar progresivamente
    if not first_agg:
        probe_start = search_end
        while probe_start < endTime:
            probe_end = min(probe_start + ONE_HOUR_MS, endTime)
            first_agg = get_aggregated_trades_by_time(symbol, probe_start, probe_end, limit=1)
            if first_agg:
                break
            probe_start = probe_end
        if not first_agg:
            market_logger.warning(f"No se encontraron aggTrades para {symbol} en el rango solicitado")
            return []

    first_atomic_id = first_agg[0]['f']  # campo 'f' = primer ID atómico del aggTrade

    # último aggTrade en endTime
    search_start = max(startTime, endTime - ONE_HOUR_MS)
    last_agg = get_aggregated_trades_by_time(symbol, search_start, endTime, limit=1000)

    # si la última hora no tiene datos, buscar hacia atrás
    if not last_agg:
        probe_end = search_start
        while probe_end > startTime:
            probe_start = max(probe_end - ONE_HOUR_MS, startTime)
            last_agg = get_aggregated_trades_by_time(symbol, probe_start, probe_end, limit=1000)
            if last_agg:
                break
            probe_end = probe_start
        if not last_agg:
            market_logger.warning(f"No se encontraron aggTrades para {symbol} en el rango solicitado")
            return []

    last_atomic_id = last_agg[-1]['l']  # campo 'l' = último ID atómico del aggTrade

    market_logger.info(
        f"Rango de IDs atómicos descubierto para {symbol}: {first_atomic_id} -> {last_atomic_id} "
        f"(estimado {last_atomic_id - first_atomic_id + 1} trades)")

    # paginar forward con /api/v3/historicalTrades
    all_trades = []
    current_id = first_atomic_id
    requests_cnt = 0

    while current_id <= last_atomic_id:
        batch = get_atomic_trades(symbol=symbol, fromId=current_id, limit=1000)
        requests_cnt += 1
        if not batch:
            break
        all_trades.extend(batch)
        current_id = batch[-1][id_field] + 1
        market_logger.info(f"Peticiones API atomic trades {symbol}: {requests_cnt} (ID actual: {current_id})")
        if len(batch) < 1000:
            break

    market_logger.info(f"Atomic trades {symbol}: {requests_cnt} peticiones, {len(all_trades)} trades obtenidos")

    # filtrar por timestamp [startTime, endTime] y deduplicar O(n) por 'id'
    seen = set()
    result = [t for t in all_trades
              if startTime <= t[timestamp_field] <= endTime
              and t[id_field] not in seen
              and not seen.add(t[id_field])]
    return sorted(result, key=lambda x: x[id_field])


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
    df[col] = df[timestamps_col]
    if time_zone != 'UTC':  # converts to time zone the time columns
        df[col] = convert_ms_column_to_datetime_with_zone(df, col, time_zone=time_zone)
        df[col] = df[col].apply(lambda x: convert_datetime_to_string(x))
    else:
        df[col] = df[timestamps_col].apply(lambda x: convert_milliseconds_to_utc_string(x))

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

    client = _get_panzer()
    return client.depth(symbol=symbol, limit=limit)


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
    client = _get_panzer()
    params = {'symbol': symbol} if symbol else {}
    response = client.get('/api/v3/ticker/bookTicker', params=params, weight=2 if not symbol else 1)
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
                            coin_qty: float = 1) -> float | None:
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


def convert_coin(coin: str, decimal_mode: bool, convert_to: str = 'BUSD', coin_qty: float | dd = 1, prices: dict = None) -> float | None:
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