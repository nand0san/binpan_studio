from tqdm import tqdm
from .logs import Logs
from .quest import check_minute_weight, get_response, get_server_time
from .time_helper import tick_seconds, end_time_from_start_time, start_time_from_end_time

market_logger = Logs(filename='./logs/market_logger.log', name='market_logger', info_level='DEBUG')

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
    En caso de superarse el límite, prima el start_time ante el end_time, start_time vendrá en milisegundos.

    En caso de timeStamps, limit es ignorado.

    La API redondea el startTime hasta el siguiente open de la siguiente candle. Es decir, no incluye la vela en la que
    está ese timeStamp, sino la siguiente vela del correspondiente tick_interval.

    El endTime indicado incluirá la vela en la que está ese timestamp. Vendrá en milisegundos.

    Si no se pasan timestamps, se retornan las últimas limit velas hasta el limit.
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
    Ojo que la API redondea el startTime hasta el siguiente open de la siguiente candle, excepto que coincida con un timestamp de Open

    No trato de incluir la vela que incluiría esa start timestamp porque si no, en candles la iteración daría
    error.

    Verificado en un jupyter notebook.
    """
    # curframe = inspect.currentframe()
    # calframe = inspect.getouterframes(curframe, 2)
    # market_logger.info(f"get_candles_from_start_time CALLED BY: {calframe[1][3]}")

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
    Get a list of candles for a specific symbol.
    The returned list of lists will be limited to limit value to the current candle.
    Maximum is 1000.
    """
    check_minute_weight(1)
    endpoint = '/api/v3/klines?'
    params = {'symbol': symbol,
              'interval': tick_interval,
              'limit': limit}
    return get_response(url=endpoint, params=params)


def get_agg_trades(fromId=None, symbol='BTCUSDT', limit=None, startTime=None, endTime=None):
    """
    Returns aggregated trades from id to limit or last trades if id not specified.

    Also is possible to get from starTime utc in milliseconds or until endtime

    Si se prueba con mas de 1h de trades da error 1127 y si ajustas a una hora se aplica el limit de 1000 máximo.

    Limit aplicado en modo fromId por defecto 500.
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
    """Returns aggregated trades between timestamps. It iterates over 1 hour intervals."""
    hour_ms = 60 * 60 * 1000
    if endTime - startTime < hour_ms:
        return get_agg_trades(symbol=symbol, startTime=startTime, endTime=endTime)
    else:
        trades = []
        for i in tqdm(range(startTime, endTime, hour_ms)):
            trades += get_agg_trades(symbol=symbol, startTime=i, endTime=i + hour_ms)
        return trades
