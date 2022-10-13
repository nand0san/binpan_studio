from tqdm import tqdm
from time import time
import pandas as pd
import json
from decimal import Decimal as dd

import handlers.time_helper
from .logs import Logs
from .quest import check_weight, get_response, api_raw_get
from .time_helper import tick_seconds, end_time_from_start_time, start_time_from_end_time

market_logger = Logs(filename='./logs/market_logger.log', name='market_logger', info_level='INFO')

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


###########
# Candles #
###########

def get_candles_by_time_stamps(symbol: str,
                               tick_interval: str,
                               start_time: int = None,
                               end_time: int = None,
                               limit=1000,
                               redis_client: object = None) -> list:
    """
    Calls API for candles list buy one or two timestamps, starting and ending.

    In case the limit is passed and exceeded by requested time intervals, the start_time prevails over the end_time,
    start_time must come in milliseconds from epoch.

    In case of two timeStamps as arguments, limit is ignored.

    The API rounds the startTime up to the next open of the next candle. That is, it does not include the candle in which there is
    that timeStamp, but the next candle of the corresponding tick_interval, except in case it exactly matches the value of an open
    timestamp, in which case it will include it in the return.

    The indicated endTime will include the candlestick that timestamp is on. It will come in milliseconds. It can be not a closed one if
    is open right in between the endtime timestamp.

    If no timestamps are passed, the last quantity candlesticks up to limit count are returned.

    :param str symbol: A binance valid symbol.
    :param str tick_interval: A binance valid time interval for candlesticks.
    :param int start_time: A timestamp in milliseconds from epoch.
    :param int end_time: A timestamp in milliseconds from epoch.
    :param int limit: Count of candles to ask for.
    :param bool redis_client: A redis instance of a connector.
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
            ret = redis_client.zrangebyscore(name=f"{symbol.lower()}@kline_{tick_interval}",
                                             min=start,
                                             max=end,
                                             withscores=False)

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

    # TODO: CORRECT INDENTATION?

    # descarta sobrantes
    overtime_candle_ts = handlers.time_helper.next_open_by_milliseconds(ms=end_time, tick_interval=tick_interval)
    if raw_candles:
        if type(raw_candles[0]) == list:  # if from binance
            raw_candles = [i for i in raw_candles if int(i[0]) < overtime_candle_ts]
        else:
            open_ts_key = list(raw_candles[0].keys())[0]
            raw_candles = [i for i in raw_candles if int(i[open_ts_key]) < overtime_candle_ts]

    return raw_candles


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


def parse_candles_to_dataframe(raw_response: list,
                               symbol: str,
                               tick_interval: str,
                               columns: list = None,
                               time_cols: list = ['Open time', 'Close time'],
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
    if not columns:
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote volume',
                   'Trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore']

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

    for col in df.columns:
        df[col] = pd.to_numeric(arg=df[col], downcast='integer')

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


##########
# Trades #
##########


def get_agg_trades(fromId: int = None, symbol: str = 'BTCUSDT', limit=None, startTime: int = None, endTime: int = None):
    """
    Returns aggregated trades from id to limit or last trades if id not specified. Also is possible to get from starTime utc in
    milliseconds from epoch or until endtime milliseconds from epoch.

    If it is tested with more than 1 hour of trades, it gives error 1127 and if you adjust it to one hour,
    the maximum limit of 1000 is NOT applied.

    Limit applied in fromId mode defaults to 500. Maximum is 1000.

    :param int fromId: An aggregated trade id.
    :param str symbol: A binance valid symbol.
    :param int limit: Count of trades to ask for.
    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :return list: Returns a list from the Binance API

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

    if fromId and not startTime and not endTime:
        query = {'symbol': symbol, 'limit': limit, 'fromId': fromId}
    elif startTime and endTime:  # Limited to one hour by api
        query = {'symbol': symbol, 'limit': limit, 'startTime': startTime, 'endTime': endTime}

    elif startTime or endTime:
        # Limited to one hour by api
        query = {'symbol': symbol, 'limit': limit}
        if startTime and not endTime:
            query.update({'startTime': startTime})
            query.update({'endTime': end_time_from_start_time(startTime=startTime,
                                                              limit=1,
                                                              tick_interval='1h')})
        if endTime and not startTime:
            query.update({'endTime': endTime})
            query.update({'startTime': start_time_from_end_time(endTime=endTime,
                                                                limit=1,
                                                                tick_interval='1h')})
    else:  # last ones
        query = {'symbol': symbol, 'limit': limit}
    if not limit:
        del query['limit']
    return get_response(url=endpoint, params=query)


def get_historical_aggregated_trades(symbol: str,
                                     startTime: int,
                                     endTime: int):
    """
    Returns aggregated trades between timestamps. It iterates over 1 hour intervals to avoid API one hour limit.

    :param int startTime: A timestamp in milliseconds from epoch.
    :param int endTime: A timestamp in milliseconds from epoch.
    :param str symbol: A binance valid symbol.
    :return list: Returns a list from the Binance API

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
    hour_ms = 60 * 60 * 1000
    if endTime - startTime < hour_ms:
        return get_agg_trades(symbol=symbol, startTime=startTime, endTime=endTime)
    else:
        trades = []
        for i in tqdm(range(startTime, endTime, hour_ms)):
            trades += get_agg_trades(symbol=symbol, startTime=i, endTime=i + hour_ms)
        return trades


#############
# Orderbook #
#############


def get_order_book(symbol='BTCUSDT', limit=5000) -> dict:
    """
    Returns a dictionary with timestamp, bids and asks. Bids and asks are a list of lists with strings representing price and quantity.
    :param str symbol: A Binance valid symbol.
    :param int limit: Max is 5000. Default 5000.
    :return dict:
    """

    endpoint = '/api/v3/depth?'
    check_weight(limit // 100 or 1, endpoint=endpoint)

    query = {'symbol': symbol, 'limit': limit}
    return get_response(url=endpoint, params=query)


####################
# coin conversions #
####################


def intermediate_conversion(coin: str,
                            decimal_mode: bool,
                            prices: dict = None,
                            try_coin: str = 'BTC',
                            coin_qty: float = 1) -> float or None:
    """
    Uses an intermediate symbol for conversion.

    :param str coin: A binance coin.
    :param bool decimal_mode: It flags to work in decimal mode.
    :param dict prices: BinPan prices dict
    :param str try_coin:
    :param float coin_qty:
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
