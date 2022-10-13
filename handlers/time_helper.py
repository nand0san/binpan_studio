from datetime import datetime
import pandas as pd
import pytz
from time import time

from .quest import get_server_time
from .logs import Logs

time_logger = Logs(filename='./logs/time_helpers.log', name='time_helpers', info_level='INFO')

tick_seconds = {'1m': 60, '3m': 60 * 3, '5m': 5 * 60, '15m': 15 * 60, '30m': 30 * 60, '1h': 60 * 60, '2h': 60 * 60 * 2,
                '4h': 60 * 60 * 4, '6h': 60 * 60 * 6, '8h': 60 * 60 * 8, '12h': 60 * 60 * 12, '1d': 60 * 60 * 24,
                '3d': 60 * 60 * 24 * 3, '1w': 60 * 60 * 24 * 7, '1M': 60 * 60 * 24 * 30}

tick_interval_values = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']


# time control functions

def convert_utc_ms_column_to_time_zone(df: pd.DataFrame, col: str, time_zone='Europe/Madrid') -> pd.Series:
    df[col] = pd.to_datetime(df[col], unit='ms')
    return df[col].dt.tz_localize('utc').dt.tz_convert(time_zone)


def convert_datetime_to_string(dt: datetime) -> str:
    """Convierte datetime tanto utc como local. No cambia zona horaria"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def convert_datetime_to_milliseconds(dt: datetime, timezoned: str = None) -> float:
    if not timezoned:
        epoch = datetime.utcfromtimestamp(0)
    else:
        time_zone = pytz.UTC
        # epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
        epoch = time_zone.localize(datetime.utcfromtimestamp(0))
    return (dt - epoch).total_seconds() * 1000.0


def convert_string_to_milliseconds(ts: str, timezoned: str = None) -> int:
    dt = convert_string_to_datetime(ts=ts, timezoned=timezoned)
    return int(convert_datetime_to_milliseconds(dt=dt, timezoned=timezoned))


def datetime_utc_to_milliseconds(dt: datetime):
    dt = dt.replace(tzinfo=pytz.UTC)
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
    return (dt - epoch).total_seconds() * 1000.0


def end_time_from_start_time(startTime: int, limit=1000, tick_interval='1h') -> int:
    return startTime + (limit * tick_seconds[tick_interval] * 1000)


def start_time_from_end_time(endTime: int, limit=1000, tick_interval='1h') -> int:
    return endTime - (limit * tick_seconds[tick_interval] * 1000)


def convert_milliseconds_to_local_timezone_string(ms: int) -> str:
    seconds = int(ms) / 1000
    return str(datetime.fromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f'))


def convert_milliseconds_to_utc_string(ms: int) -> str:
    seconds = int(ms) / 1000
    return str(datetime.utcfromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f'))


def convert_milliseconds_to_utc_datetime(ms: int) -> datetime:
    seconds = int(ms) / 1000
    return datetime.utcfromtimestamp(seconds)


def convert_milliseconds_to_time_zone_datetime(ms: int, timezoned: str = None) -> datetime:
    seconds = int(ms) / 1000
    dt = datetime.utcfromtimestamp(seconds)
    utc_tz = pytz.UTC
    utc_datetime = utc_tz.localize(dt)
    time_zone = pytz.timezone(timezoned)
    return utc_datetime.astimezone(time_zone)


def convert_milliseconds_to_str(ms: int, timezoned: str = None) -> str:
    dt = convert_milliseconds_to_time_zone_datetime(ms=ms, timezoned=timezoned)
    return convert_datetime_to_string(dt)


def utc_start_of_day_ms() -> int:
    now = get_server_time()
    dt = convert_milliseconds_to_utc_datetime(now).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(convert_datetime_to_milliseconds(dt))


def utc_ms_day_begin_end(ts: int) -> (int, int):
    """Calcula en horario utc el timestamp del comienzo del dia y su final del timestamp pasado como argumento."""
    day_ini_df = convert_milliseconds_to_utc_datetime(ts).replace(hour=0, minute=0, second=0, microsecond=0)
    day_ini_ms = convert_utc_datetime_to_milliseconds(day_ini_df)
    day_end_ms = day_ini_ms + (24 * 60 * 60 * 1000) - 1
    return int(day_ini_ms), int(day_end_ms)


def split_time_interval_in_full_days(ts_ini: int, ts_end: int) -> list:
    """Dado un intervalo mediante dos timestamps, devuelve un alista de tuplas con el inicio y final de cada día completo
     tocado por el intervalo input."""
    ini_first_day, end_first_day_ = utc_ms_day_begin_end(ts_ini)
    ini_last_day_, end_last_day = utc_ms_day_begin_end(ts_end)
    day_ms = 24 * 60 * 60 * 1000
    days = int(ceil_division((end_last_day - ini_first_day), day_ms))
    ret = []
    for i in range(days):
        day_ts_ini = ini_first_day + i * day_ms
        day_ts_end = day_ts_ini + day_ms - 1
        ret.append((int(day_ts_ini), int(day_ts_end)))
    return ret


def time_interval(tick_interval: str,
                  limit: int = 1000,
                  start: int = None,
                  end: int = None) -> tuple:
    """
    Obtain a timestamp based on ticks intervals from a start or an end timestamp, based on limit.

    If no start or end timestamp passed, then use current utc timestamp in milliseconds and limit.

    :param str tick_interval: A binance valid tick interval

    :param int start: A timestamp in milliseconds.
    :param int end: A timestamp in milliseconds.
    :param int limit: Ticks limit. Not applied if start and end passed. Default is 1000

    :return: A tuple with the start and end timestamps.
    """
    total_interval_ms = tick_seconds[tick_interval] * 1000 * limit
    if not start and not end:
        end = utc()
        start = end - total_interval_ms
    elif not end and start:
        end = int(start) + total_interval_ms
    elif not start and end:
        start = int(end) - total_interval_ms
    return int(start), int(end)


def ceil_division(a: float, b: float) -> int:
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return int(-(a // -b))


def ticks_between_timestamps(start: int,
                             end: int,
                             tick_interval: str) -> int:
    """
    Dados dos timestamps en milliseconds y un tick interval te dice cuantas velas entran ahi.

    Se aplica:
        - si el start es un open se toma la vela en la que abre, si no, se toma la siguiente vela.
        - si el end es un open se toma la vela de ese open también, si no, igualmente, la vela que contiene al end.

    El limit no se tiene en cuenta aquí, previamente se habrá tenido en cuenta.
    """
    start_open = open_from_milliseconds(ms=start, tick_interval=tick_interval)
    if start_open != start:
        start_open = next_open_by_milliseconds(ms=start, tick_interval=tick_interval)
    end_close = close_from_milliseconds(end, tick_interval=tick_interval)
    ret = ceil_division(end_close - start_open, tick_seconds[tick_interval]*1000)
    return ret


def get_ticks_to_timestamp_utc(ticks_interval: str, timestamp: int = None):
    """Retorna cuantas velas se requieren hasta el timestamp pasado hasta ahora"""
    now = get_server_time()
    tick_ms = tick_seconds[ticks_interval] * 1000
    interval_ms = now - timestamp
    return interval_ms // tick_ms


def convert_string_to_datetime(ts: str, timezoned: str = None):
    try:
        ret = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        ret = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    if timezoned:
        mytz = pytz.timezone(timezoned)
        return mytz.localize(ret)
    else:
        return ret


def convert_utc_datetime_to_milliseconds(dt):
    epoch = datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0


def utc() -> int:
    """
    Returns UTC time in milliseconds from epoch.
    :return: int
    """
    return int(datetime.now().timestamp() * 1000)


def utc_datetime() -> datetime:
    return datetime.now(pytz.utc)


# Checkers and waiters

def check_tick_interval(tick_interval: str) -> str:
    """
    Checks if argument is a Binance valid tick interval for candles.

    :param str tick_interval: A string, maybe, binance tick interval well formatted.

    """
    if not (tick_interval.lower() in tick_interval_values):
        if not (tick_interval.upper() in tick_interval_values):
            raise Exception(f"BinPan Error on tick_interval: {tick_interval} not in "
                            f"expected API intervals.\n{tick_interval_values}")
        else:
            return tick_interval.upper()
    else:
        return tick_interval.lower()


def detect_tick_interval(data: pd.DataFrame) -> str:
    """Retorna el tick interval de un dataframe"""
    ts_a = data.iloc[0]['Open timestamp']
    ts_b = data.iloc[1]['Open timestamp']
    seconds = (ts_b - ts_a) // 1000
    return list(tick_seconds.keys())[list(tick_seconds.values()).index(seconds)]


def next_open_utc(tick_interval: str) -> int:
    """Calcula el timestamp del next open para un tick_interval"""
    utc_ms = get_server_time()
    units = (utc_ms // (tick_seconds[tick_interval] * 1000))
    last_open_ms = units * tick_seconds[tick_interval] * 1000
    return last_open_ms + (tick_seconds[tick_interval] * 1000)


def next_open_by_milliseconds(ms: int, tick_interval: str) -> int:
    """Calcula el timestamp del next open para un tick_interval."""
    units = (int(ms) // (tick_seconds[tick_interval] * 1000))
    last_open_ms = units * tick_seconds[tick_interval] * 1000
    return last_open_ms + (tick_seconds[tick_interval] * 1000)


def open_from_milliseconds(ms: int, tick_interval: str) -> int:
    """Retorna el open de la vela definida que contiene ese timestamp.
    Si coincide el timestamp con un open, calcula y devuelve ese mismo timestamp."""
    units = (int(ms) // (tick_seconds[tick_interval] * 1000))
    return units * tick_seconds[tick_interval] * 1000


def close_from_milliseconds(ms: int, tick_interval: str) -> int:
    """Retorna el close de la vela definida que contiene ese timestamp.
    Si coincide el timestamp con un open, calcula el final."""
    units = (int(ms) // (tick_seconds[tick_interval] * 1000))
    units += 1
    return (units * tick_seconds[tick_interval] * 1000) - 1000


def wait_seconds_until_next_minute():
    # server_time = binpan_modules.api_control.api.get_server_time()
    server_time = int(time() * 1000)  # ahorro de llamadas al server, mejorada sincro de tiempo en windows
    now = datetime.fromtimestamp(server_time / 1000.0).replace(tzinfo=pytz.UTC)
    return 60 - now.second
