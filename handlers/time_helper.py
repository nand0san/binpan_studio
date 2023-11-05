from datetime import datetime
import pytz
from time import time
from typing import Tuple, List
import pandas as pd

from .logs import Logs

time_logger = Logs(filename='./logs/time_helpers.log', name='time_helpers', info_level='INFO')

tick_seconds = {'1m': 60, '3m': 60 * 3, '5m': 5 * 60, '15m': 15 * 60, '30m': 30 * 60, '1h': 60 * 60, '2h': 60 * 60 * 2,
                '4h': 60 * 60 * 4, '6h': 60 * 60 * 6, '8h': 60 * 60 * 8, '12h': 60 * 60 * 12, '1d': 60 * 60 * 24,
                '3d': 60 * 60 * 24 * 3, '1w': 60 * 60 * 24 * 7, '1M': 60 * 60 * 24 * 30}
pandas_freq_tick_interval = {'1m': '1T',
                             '3m': '3T',
                             '5m': '5T',
                             '15m': '15T',
                             '30m': '30T',
                             '1h': '1H',
                             '2h': '2H',
                             '4h': '4H',
                             '6h': '6H',
                             '8h': '8H',
                             '12h': '12H',
                             '1d': '1D',
                             '3d': '3D',
                             '1w': '1W',
                             '1M': '1M'}

tick_interval_values = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']


##########################
# time control functions #
##########################


def convert_ms_column_to_datetime_with_zone(df: pd.DataFrame, col: str, time_zone='Europe/Madrid', ambiguous='infer') -> pd.Series:
    """
    Replace a column from milliseconds to datetime with time zone.

    :param df: A pandas dataframe.
    :param col: Name of the column to replace.
    :param time_zone: A time zone like 'Europe/Madrid'
    :param ambiguous: A string or list to resolve ambiguous times during DST transitions. Default is 'infer'.
    :return: Modified dataframe.
    """
    df[col] = pd.to_datetime(df[col], unit='ms')

    # Add timezone check
    if df[col].dt.tz is None:
        return df[col].dt.tz_localize('utc', ambiguous=ambiguous).dt.tz_convert(time_zone)
    else:
        return df[col].dt.tz_convert(time_zone)


def convert_datetime_to_string(dt) -> str:
    """
    Converts a datetime to a string.

    :param dt: A datetime object.
    :return: A string with a date and time. Example: '2021-01-01 00:00:00.000'
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def convert_datetime_utc2ms(date_object: datetime) -> int:
    """
    Convert datetime object to milliseconds. Example: datetime.datetime(2023, 10, 10, 11, 52, tzinfo=datetime.timezone.utc)

    :param date_object: A datetime object.
    :return: A Unix timestamp in milliseconds.
    """
    return int(date_object.timestamp() * 1000)


def convert_datetime_to_milliseconds(dt: datetime, timezoned: str = None) -> float:
    """
    Converts a datetime to milliseconds. If string contains a timezone, it will be removed when passing timezoned.

    :param dt: A datetime object.
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: A Unix timestamp in milliseconds.
    """
    if not timezoned:
        epoch = datetime.utcfromtimestamp(0)
    else:
        time_zone = pytz.UTC
        # epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
        epoch = time_zone.localize(datetime.utcfromtimestamp(0))
    return (dt - epoch).total_seconds() * 1000.0


def convert_string_to_milliseconds(ts: str, timezoned: str = None) -> int:
    """
    Converts a string to milliseconds. If string contains a timezone, it will be removed when passing timezoned.

    :param ts: A string with a date and time. Example: '2021-01-01 00:00:00.000'
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: Returns a timestamp in milliseconds.
    """
    if "+" in ts and timezoned:
        ts = ts.split("+")[0]
    dt = convert_string_to_datetime(ts=ts, timezoned=timezoned)
    return int(convert_datetime_to_milliseconds(dt=dt, timezoned=timezoned))


def datetime_utc_to_milliseconds(dt: datetime):
    """
    Convert datetime object to milliseconds. Example: datetime.datetime(2023, 10, 10, 11, 52, tzinfo=datetime.timezone.utc)

    :param dt: A datetime object.
    :return: A Unix timestamp in milliseconds.
    """
    dt = dt.replace(tzinfo=pytz.UTC)
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
    return (dt - epoch).total_seconds() * 1000.0


def end_time_from_start_time(startTime: int, limit=1000, tick_interval='1h') -> int:
    """
    Calculates end time by tick interval and limit.

    :param int startTime: A timestamp in milliseconds.
    :param int limit: Klines. Maximum is 1000.
    :param str tick_interval: Kline interval.
    :return int: Endtime timestamp in milliseconds.
    """
    return startTime + (limit * tick_seconds[tick_interval] * 1000)


def start_time_from_end_time(endTime: int, limit=1000, tick_interval='1h') -> int:
    """
    Calculates startime from a timestamp and a limit of klines before.

    :param int endTime: Timestamp in ms.
    :param int limit: API klines limit is 1000.
    :param str tick_interval: Kline interval.
    :return int: Timestamp in ms.
    """
    return endTime - (limit * tick_seconds[tick_interval] * 1000)


def convert_milliseconds_to_local_timezone_string(ms: int) -> str:
    """
    Converts a timestamp in milliseconds to a string.

    :param ms: A timestamp in milliseconds.
    :return: A string with a date and time. Example: '2021-01-01 00:00:00.000'
    """
    seconds = int(ms) / 1000
    return str(datetime.fromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f'))


def convert_milliseconds_to_utc_string(ms: int) -> str:
    """
    Converts a timestamp in milliseconds to a string.

    :param ms: A timestamp in milliseconds.
    :return: A string with a date and time. Example: '2021-01-01 00:00:00.000'
    """
    seconds = int(ms) / 1000
    return str(datetime.utcfromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f'))


def convert_milliseconds_to_utc_datetime(ms: int) -> datetime:
    """
    Converts a timestamp in milliseconds to a datetime.

    :param ms: A timestamp in milliseconds.
    :return: A datetime object. Example: datetime.datetime(2021, 1, 1, 0, 0)
    """
    seconds = int(ms) / 1000
    return datetime.utcfromtimestamp(seconds)


def convert_milliseconds_to_time_zone_datetime(ms: int, timezoned: str = None) -> datetime:
    """
    Converts a timestamp in milliseconds to a datetime. If string contains a timezone, it will be removed when passing timezoned.

    :param ms: A timestamp in milliseconds.
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: A datetime object. Example: datetime.datetime(2021, 1, 1, 0, 0)
    """
    seconds = int(ms) / 1000
    dt = datetime.utcfromtimestamp(seconds)
    utc_tz = pytz.UTC
    utc_datetime = utc_tz.localize(dt)
    time_zone = pytz.timezone(timezoned)
    return utc_datetime.astimezone(time_zone)


def convert_milliseconds_to_str(ms: int, timezoned: str) -> str:
    """
    Converts a timestamp in milliseconds to a string. If string contains a timezone, it will be removed when passing timezoned.

    :param ms: A timestamp in milliseconds.
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: A string with a date and time. Example: '2021-01-01 00:00:00.000'
    """
    dt = convert_milliseconds_to_time_zone_datetime(ms=ms, timezoned=timezoned)
    return convert_datetime_to_string(dt)


def utc_start_of_day_ms() -> int:
    """
    Returns UTC time in milliseconds from epoch for the start of the day.

    :return: A timestamp in milliseconds.
    """
    # now = get_server_time()
    now = int(time() * 1000)
    dt = convert_milliseconds_to_utc_datetime(now).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(convert_datetime_to_milliseconds(dt))


def ms_day_begin_end(ts: int) -> (int, int):
    """
    Given a timestamp in milliseconds, it returns the start and end timestamps of the day.

    :param ts: A timestamp in milliseconds.
    :return: A tuple with the start and end timestamps of the day.
    """
    day_ini_df = convert_milliseconds_to_utc_datetime(ts).replace(hour=0, minute=0, second=0, microsecond=0)
    day_ini_ms = convert_utc_datetime_to_milliseconds(day_ini_df)
    day_end_ms = day_ini_ms + (24 * 60 * 60 * 1000) - 1
    return int(day_ini_ms), int(day_end_ms)


def split_time_interval_in_full_days(ts_ini: int, ts_end: int) -> list:
    """
    Given two timestamps in milliseconds, it returns a list of tuples with the start and end timestamps of each day in the interval.

    :param ts_ini: A timestamp in milliseconds.
    :param ts_end: A timestamp in milliseconds.
    :return: It returns a list of tuples with the start and end timestamps of each day in the interval.
    """

    ini_first_day, end_first_day_ = ms_day_begin_end(ts_ini)
    ini_last_day_, end_last_day = ms_day_begin_end(ts_end)
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
                  start_time: int = None,
                  end_time: int = None) -> Tuple[int, int]:
    """
    Obtain a timestamp based on ticks intervals from a start or an end timestamp, based on limit.

    If no start or end timestamp passed, then use current utc timestamp in milliseconds and limit.

    :param str tick_interval: A binance valid tick interval
    :param int start_time: A timestamp in milliseconds.
    :param int end_time: A timestamp in milliseconds.
    :param int limit: Ticks limit. Not applied if start and end passed. Default is 1000
    :return: A tuple with the start and end timestamps.

    """
    total_interval_ms = int(tick_seconds[tick_interval] * 1000 * limit)
    if not start_time and not end_time:
        end_time = int(time() * 1000)
        start_time = end_time - total_interval_ms
    elif not end_time and start_time:
        end_time = int(start_time) + total_interval_ms
    elif not start_time and end_time:
        start_time = int(end_time) - total_interval_ms
    return start_time, end_time


def ceil_division(a: float, b: float) -> int:
    """
    Better not to use //.

    From: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python

    :param float a: A value.
    :param float b: Other value.
    :return int: Returns integer from ceil division.
    """
    return int(-(a // -b))


def ticks_between_timestamps(start: int,
                             end: int,
                             tick_interval: str) -> int:
    """
    Given two timestamps in milliseconds and a tick interval, it returns the number of ticks between them.

    It applies:
        - if the start is an open, the candle in which it opens is taken, if not, the next candle is taken.
        - if the end is an open, the candle of that open is also taken, if not, likewise, the candle that contains the end.

    Limit is not taken into account here, it will have been taken into account previously.

    :param start: A timestamp in milliseconds.
    :param end: A timestamp in milliseconds.
    :param tick_interval: A tick interval like '1m', '1h', etc.
    :return: Returns the number of ticks between them.

    """
    start_open = open_from_milliseconds(ms=start, tick_interval=tick_interval)
    if start_open != start:
        start_open = next_open_by_milliseconds(ms=start, tick_interval=tick_interval)
    end_close = close_from_milliseconds(end, tick_interval=tick_interval)
    ret = ceil_division(end_close - start_open, tick_seconds[tick_interval] * 1000)
    return ret


def get_ticks_to_timestamp_utc(ticks_interval: str, timestamp: int = None) -> int:
    """
    Returns the number of ticks to the next timestamp in milliseconds.

    :param ticks_interval: A tick interval like '1m', '1h', etc.
    :param timestamp: A timestamp in milliseconds. If None, it will use current time.
    :return: Returns the number of ticks to the next timestamp in milliseconds.
    """
    # now = get_server_time()
    now = int(time() * 1000)
    tick_ms = tick_seconds[ticks_interval] * 1000
    interval_ms = now - timestamp
    return interval_ms // tick_ms


def convert_string_to_datetime(ts: str, timezoned: str = None) -> datetime:
    """
    Converts a string to datetime. If string contains a timezone, it will be removed when passing timezoned.

    :param ts: A string with a date and time. Example: '2021-01-01 00:00:00.000'
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: Returns a datetime object.Example: datetime.datetime(2021, 1, 1, 0, 0)
    """
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
    """
    Convert datetime object to milliseconds. Example: datetime.datetime(2023, 10, 10, 11, 52, tzinfo=datetime.timezone.utc)

    :param dt: A datetime object.
    :return: A Unix timestamp in milliseconds.
    """
    epoch = datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0


def utc() -> int:
    """
    Returns UTC time in milliseconds from epoch.

    :return: A timestamp in milliseconds.
    """
    return int(datetime.now().timestamp() * 1000)


def utc_datetime() -> datetime:
    """
    Returns UTC datetime.

    :return: A datetime object.
    """
    return datetime.now(pytz.utc)


# Checkers and waiters

def check_tick_interval(tick_interval: str) -> str:
    """
    Checks if argument is a Binance valid tick interval for candles.

    :param str tick_interval: A string, maybe, binance tick interval well formatted.
    :return: A string with the tick interval.
    """
    if not tick_interval in tick_interval_values:
        raise Exception(f"BinPan Error on tick_interval: {tick_interval} not in "
                        f"expected API intervals.\n{tick_interval_values}")
    return tick_interval


def detect_tick_interval(data: pd.DataFrame) -> str:
    """
    Detects tick interval from a dataframe with a column named 'Open timestamp'.

    :param data: A pandas dataframe with a column named 'Open timestamp'.
    :return: A string with the tick interval detected. Example: '1m', '1h', etc.
    """
    ts_a = data.iloc[0]['Open timestamp']
    ts_b = data.iloc[1]['Open timestamp']
    seconds = (ts_b - ts_a) // 1000
    return list(tick_seconds.keys())[list(tick_seconds.values()).index(seconds)]


def next_open_utc(tick_interval: str) -> int:
    """
    Calculates the next open timestamp in milliseconds for a tick interval.

    :param tick_interval: A tick interval like '1m', '1h', etc.
    :return: It returns the next open timestamp in milliseconds.
    """
    # utc_ms = get_server_time()
    utc_ms = int(time() * 1000)
    units = (utc_ms // (tick_seconds[tick_interval] * 1000))
    last_open_ms = units * tick_seconds[tick_interval] * 1000
    return last_open_ms + (tick_seconds[tick_interval] * 1000)


def next_open_by_milliseconds(ms: int, tick_interval: str) -> int:
    """
    Calculates the next open timestamp in milliseconds for a tick interval.

    :param ms:  A timestamp in milliseconds.
    :param tick_interval: A tick interval like '1m', '1h', etc.
    :return: It returns the next open timestamp in milliseconds.
    """

    units = (int(ms) // (tick_seconds[tick_interval] * 1000))
    last_open_ms = units * tick_seconds[tick_interval] * 1000
    return last_open_ms + (tick_seconds[tick_interval] * 1000)


def open_from_milliseconds(ms: int, tick_interval: str) -> int:
    """
    Returns the open timestamp in milliseconds for a tick interval.

    :param ms: A timestamp in milliseconds.
    :param tick_interval: A tick interval like '1m', '1h', etc.
    :return: It returns the open timestamp in milliseconds for that tick interval in that timestamp.
    """

    units = (int(ms) // (tick_seconds[tick_interval] * 1000))
    return units * tick_seconds[tick_interval] * 1000


def close_from_milliseconds(ms: int, tick_interval: str) -> int:
    """
    Returns the close timestamp in milliseconds for a tick interval.

    :param ms: A timestamp in milliseconds.
    :param tick_interval: A tick interval like '1m', '1h', etc.
    :return: Returns the close timestamp in milliseconds for that tick interval in that timestamp.
    """

    units = (int(ms) // (tick_seconds[tick_interval] * 1000))
    units += 1
    return (units * tick_seconds[tick_interval] * 1000) - 1000


def wait_seconds_until_next_minute():
    """
    Waits until the next minute.

    :return: It returns the seconds to wait until the next minute.
    """
    # server_time = binpan_modules.api_control.api.get_server_time()
    server_time = int(time() * 1000)  # ahorro de llamadas al server, mejorada sincro de tiempo en windows
    now = datetime.fromtimestamp(server_time / 1000.0).replace(tzinfo=pytz.UTC)
    return 60 - now.second


def calculate_iterations(start_time: int, end_time: int, tick_interval_ms: int, limit: int = 1000) -> int:
    """
    Calculate the number of iterations required to retrieve data within the given time range considering the API limit inclusively.

    When timestamps in the middle of a kline, it will expand the time range to include the whole kline from opens including both start and
     end timestamps.

    :param int start_time: Start timestamp (milliseconds) of the time range.
    :param int end_time: End timestamp (milliseconds) of the time range.
    :param int tick_interval_ms: Kline tick interval in milliseconds.
    :param int limit: API limit for the number of klines in a single request (default: 1000).
    :return int: The number of iterations required to retrieve data within the given time range.
    """
    start = (start_time // tick_interval_ms) * tick_interval_ms
    end = -(end_time // -tick_interval_ms) * tick_interval_ms
    klines = (end - start) / tick_interval_ms
    assert int(klines) == klines, f"Error in calculate_iterations: klines is not an integer: {klines}"
    iterations = -(klines // -limit)
    assert int(iterations) == iterations, f"Error in calculate_iterations: iterations is not an integer: {iterations}"
    return int(iterations)


def infer_frequency_and_set_index(data: pd.DataFrame, timestamp_column: str = 'Open timestamp',
                                  timezone: str = None) -> pd.DataFrame:
    """
    Infers the frequency of the DataFrame based on the 'Open timestamp' column and sets it as the DataFrame index.

    :param pd.DataFrame data: Input DataFrame with a column containing timestamps in milliseconds.
    :param str timestamp_column: Name of the column in the DataFrame that contains the timestamps. Default is 'Open timestamp'.
    :param str timezone: The timezone to use when converting timestamps to datetime. Default is None to get from the dataframe data.
    :return pd.DataFrame: DataFrame with the timestamp column set as the index and the frequency inferred and set.
    :raises ValueError: If the DataFrame index is not of type datetime.
    """
    timezone_from_data = data.index.name.split()[-1]
    if not timezone:
        timezone = timezone_from_data

    df = data.copy(deep=True)
    # Convertir el timestamp de milisegundos a datetime y establecerlo como índice
    df = df.set_index(pd.to_datetime(df[timestamp_column], unit='ms').dt.tz_localize('UTC').dt.tz_convert(timezone))

    # Verificar que el índice es de tipo datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo datetime.")
    # Inferir la frecuencia
    inferred_freq = pd.infer_freq(df.index)

    # Si se pudo inferir una frecuencia, establecerla en el índice
    if inferred_freq:
        df = df.asfreq(inferred_freq)
    else:
        print("No se pudo inferir una frecuencia para el índice.")
    return df


def get_dataframe_time_index_ranges(data: pd.DataFrame, interval='30T') -> List[tuple]:
    """
    Divides a DataFrame into time ranges of a specified interval. Complete intervals are used, but the first interval is the one with
     the rest.

    :param pd.DataFrame data: A dataframe with a datetime index.
    :param interval: Interval expressed as a Pandas frequency string. Default is '30T' (30 minutes).
    :return list: A list of tuples containing the start and end time of each interval.

    Example:
    """
    df_ = data.sort_index(ascending=False)  # reversed order
    start_time = df_.index.min()
    end_time = df_.index.max()
    time_ranges = []

    current_time = end_time

    while current_time > start_time:
        range_start = max(current_time - pd.Timedelta(interval), start_time)
        time_ranges.append((range_start, current_time))
        current_time -= pd.Timedelta(interval)

    return sorted(time_ranges)


def remove_initial_included_ranges(time_ranges, initial_minutes) -> List[tuple]:
    """
    Remove the time ranges that are completely included in the initial period.

    :param time_ranges: A list of tuples containing the start and end time of each interval.
    :param initial_minutes: A quantity of minutes that defines the initial period.
    :return: A list of tuples containing the start and end time of each interval, without the ranges that are completely included in the
     initial period.
    """
    if not time_ranges:
        return []

    initial_end_time = time_ranges[0][0] + pd.Timedelta(minutes=initial_minutes)

    modified_time_ranges = [next((start_time, end_time) for start_time, end_time in time_ranges if end_time > initial_end_time)]

    for start_time, end_time in time_ranges[1:]:
        # Si el tiempo de inicio del rango de tiempo actual es anterior al tiempo de finalización del último rango de tiempo en la lista
        # modificada,
        # entonces el rango de tiempo actual está completamente incluido en un rango de tiempo desde el comienzo del primer intervalo que
        # aparece,
        # por lo que lo saltamos
        if start_time < modified_time_ranges[-1][1]:
            continue
        modified_time_ranges.append((start_time, end_time))
    return modified_time_ranges


def adjust_timestamp_unit_nano_or_ms(ts):
    # Contar el número de dígitos
    num_digits = len(str(ts))

    # Si está en milisegundos, convertir a nanosegundos
    if num_digits <= 13:
        ts *= 1_000_000

    return ts
