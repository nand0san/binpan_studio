from tqdm import tqdm
from .logs import Logs
from .quest import check_minute_weight, get_response, get_server_time
from .time_helper import tick_seconds, end_time_from_start_time, start_time_from_end_time

market_logger = Logs(filename='./logs/market_logger.log', name='market_logger', info_level='INFO')

base_url = 'https://api.binance.com'


# ###################################
# # API market
# ###################################


def get_candles_by_time_stamps(start_time: int = None,
                               end_time: int = None,
                               symbol='BTCUSDT',
                               tick_interval='1d',
                               limit=None) -> list:
    """
    Calls API for candles list buy one or two timestamps, starting and ending.

    In case the limit is exceeded, the start_time prevails over the end_time, start_time must come in milliseconds from epoch.

    In case of two timeStamps as arguments, limit is ignored.

    The API rounds the startTime up to the next open of the next candle. That is, it does not include the candle in which there is
    that timeStamp, but the next candle of the corresponding tick_interval, except in case it exactly matches the value of an open
    timestamp, in which case it will include it in the return.

    The indicated endTime will include the candlestick that timestamp is on. It will come in milliseconds. It can be not a closed one if
    is open right in between the endtime timestamp.

    If no timestamps are passed, the last quantity candlesticks up to limit count are returned.

    :param int start_time: A timestamp in milliseconds from epoch.
    :param int end_time: A timestamp in milliseconds from epoch.
    :param str symbol: A binance valid symbol.
    :param str tick_interval: A binance valid time interval for candlesticks.
    :param int limit: Count of candles to ask for.
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

    now = get_server_time()
    if end_time and end_time > now:
        end_time = None
        # end_time = min(end_time, now)
    check_minute_weight(1)
    endpoint = '/api/v3/klines?'

    if not start_time and end_time:
        start_time = end_time - (limit * tick_seconds[tick_interval] * 1000)

    elif start_time and not end_time:
        end_time = start_time + (limit * tick_seconds[tick_interval] * 1000) - 1

    params = {'symbol': symbol,
              'interval': tick_interval,
              'startTime': start_time,
              'endTime': end_time,
              'limit': limit}

    params = {k: v for k, v in params.items() if v}
    return get_response(url=endpoint, params=params)


def get_candles_from_start_time(start_time: int,
                                symbol: str = 'BTCUSDT',
                                tick_interval: str = '1d',
                                limit: int = 1000) -> list:
    """
    Calls API for candles list from one timestamp.

    The API rounds the startTime up to the next open of the next candle. That is, it does not include the candle in which there is
    that timeStamp, but the next candle of the corresponding tick_interval, except in case it exactly matches the value of an open
    timestamp, in which case it will include it in the return.

    :param int start_time: A timestamp in milliseconds from epoch.
    :param str symbol: A binance valid symbol.
    :param str tick_interval: A binance valid time interval for candlesticks.
    :param int limit: Count of candles to ask for.
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
    market_logger.info(f"Candles for: {locals()}")
    check_minute_weight(1)
    endpoint = '/api/v3/klines?'
    params = {'symbol': symbol,
              'interval': tick_interval,
              'startTime': start_time,
              'limit': limit}
    return get_response(url=endpoint, params=params)


def get_last_candles(symbol: str = 'BTCUSDT',
                     tick_interval: str = '1d',
                     limit: int = 1000) -> list:
    """
    Calls API for candles list from one timestamp for a specific symbol.

    The returned list of lists will be limited to limit quantity until the current candle.

    Maximum is 1000.

    :param str symbol: A binance valid symbol.
    :param str tick_interval: A binance valid time interval for candlesticks.
    :param int limit: Count of candles to ask for.
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
    check_minute_weight(1)
    endpoint = '/api/v3/klines?'
    params = {'symbol': symbol,
              'interval': tick_interval,
              'limit': limit}
    return get_response(url=endpoint, params=params)


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
    check_minute_weight(1)
    endpoint = '/api/v3/aggTrades?'
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
