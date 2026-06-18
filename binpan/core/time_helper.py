"""

Time helper functions.

Kline boundary functions delegated to kline-timestamp library.
Generic datetime/string/ms conversions kept here.

"""

from datetime import datetime, timezone
import pytz
import pandas as pd
from kline_timestamp import KlineTimestamp

from .logs import LogManager

time_logger = LogManager(filename='./logs/time_helpers.log', name='time_helpers', info_level='INFO')

# Nota: '1M' usa 30 días como aproximación. Para duración exacta por mes usar KlineTimestamp.tick_ms
tick_seconds = {'1m': 60, '3m': 60 * 3, '5m': 5 * 60, '15m': 15 * 60, '30m': 30 * 60, '1h': 60 * 60, '2h': 60 * 60 * 2,
                '4h': 60 * 60 * 4, '6h': 60 * 60 * 6, '8h': 60 * 60 * 8, '12h': 60 * 60 * 12, '1d': 60 * 60 * 24,
                '3d': 60 * 60 * 24 * 3, '1w': 60 * 60 * 24 * 7, '1M': 60 * 60 * 24 * 30}

pandas_freq_tick_interval = {'1m': '1min',
                             '3m': '3min',
                             '5m': '5min',
                             '15m': '15min',
                             '30m': '30min',
                             '1h': '1h',
                             '2h': '2h',
                             '4h': '4h',
                             '6h': '6h',
                             '8h': '8h',
                             '12h': '12h',
                             '1d': '1D',
                             '3d': '3D',
                             '1w': '1W',
                             '1M': '1ME'}

tick_interval_values = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']


############################
# parse_timestamp (migrado #
# de kline-timestamp)      #
############################


def parse_timestamp(timestamp_str: str, timezone: str | pytz.BaseTzInfo | None = "Europe/Madrid") -> datetime:
    """
    Parses a timestamp string in multiple formats and returns a timezone-aware datetime.

    If the timestamp string doesn't include time zone information, it is assumed to be in UTC.
    If timezone parameter is specified, the datetime is converted to that timezone.

    :param str timestamp_str: Timestamp string. Accepts ISO 8601 and common date/time formats.
    :param timezone: Time zone in IANA format (e.g., "Europe/Madrid") or pytz timezone object.
    :return datetime: Timezone-aware datetime object.
    """
    formats = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y/%m/%d %H:%M:%S%z",
        "%d-%m-%Y %H:%M:%S%z",
        "%d/%m/%Y %H:%M:%S%z",
        "%Y%m%d%H%M%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y%m%d%H%M%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y%m%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            if dt.tzinfo is not None:
                dt = dt.astimezone(pytz.utc)
            else:
                dt = dt.replace(tzinfo=pytz.utc)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Timestamp '{timestamp_str}' is not in a recognized format.")

    if timezone is None:
        pass
    elif isinstance(timezone, str):
        timezone = pytz.timezone(timezone)
    elif not (hasattr(timezone, 'localize') or hasattr(timezone, 'utcoffset')):
        raise TypeError(f"Timezone must be a string or a pytz timezone object, not {type(timezone)}")

    if timezone:
        dt = dt.astimezone(timezone)
    return dt


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
    dt_series = pd.to_datetime(df[col], unit='ms')

    if dt_series.dt.tz is None:
        return dt_series.dt.tz_localize('utc', ambiguous=ambiguous).dt.tz_convert(time_zone)
    else:
        return dt_series.dt.tz_convert(time_zone)


def convert_datetime_to_string(dt) -> str:
    """
    Converts a datetime to a string.

    :param dt: A datetime object.
    :return: A string with a date and time. Example: '2021-01-01 00:00:00'
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def convert_datetime_to_milliseconds(dt: datetime, timezoned: str = None) -> float:
    """
    Converts a datetime to milliseconds. If timezoned, assumes the datetime is in that timezone.

    :param dt: A datetime object.
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: A Unix timestamp in milliseconds.
    """
    if not timezoned:
        epoch = datetime.fromtimestamp(0, tz=timezone.utc).replace(tzinfo=None)
    else:
        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
    return (dt - epoch).total_seconds() * 1000.0


def convert_string_to_milliseconds(ts: str, timezoned: str = None) -> int:
    """
    Converts a string to milliseconds.

    :param ts: A string with a date and time. Example: '2021-01-01 00:00:00'
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: Returns a timestamp in milliseconds.
    """
    dt = convert_string_to_datetime(ts=ts, timezoned=timezoned)
    return int(convert_datetime_to_milliseconds(dt=dt, timezoned=timezoned))


def convert_milliseconds_to_utc_string(ms: int) -> str:
    """
    Converts a timestamp in milliseconds to a UTC string.

    :param ms: A timestamp in milliseconds.
    :return: A string with a date and time.
    """
    seconds = int(ms) / 1000
    return str(datetime.fromtimestamp(seconds, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f'))


def convert_milliseconds_to_time_zone_datetime(ms: int, timezoned: str = None) -> datetime:
    """
    Converts a timestamp in milliseconds to a timezone-aware datetime.

    :param ms: A timestamp in milliseconds.
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: A datetime object.
    """
    seconds = int(ms) / 1000
    utc_datetime = datetime.fromtimestamp(seconds, tz=timezone.utc)
    time_zone = pytz.timezone(timezoned)
    return utc_datetime.astimezone(time_zone)


def convert_milliseconds_to_str(ms: int, timezoned: str) -> str:
    """
    Converts a timestamp in milliseconds to a timezone-aware string.

    :param ms: A timestamp in milliseconds.
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: A string with a date and time.
    """
    dt = convert_milliseconds_to_time_zone_datetime(ms=ms, timezoned=timezoned)
    return convert_datetime_to_string(dt)


def convert_string_to_datetime(ts: str, timezoned: str = None) -> datetime:
    """
    Converts a string to datetime. If timezoned, localizes the result.

    :param ts: A string with a date and time. Example: '2021-01-01 00:00:00'
    :param timezoned: A timezone like 'Europe/Madrid'
    :return: Returns a datetime object.
    """
    try:
        ret = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            ret = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            ret = datetime.strptime(ts, '%Y-%m-%d')
    if timezoned:
        mytz = pytz.timezone(timezoned)
        return mytz.localize(ret)
    else:
        return ret


#######################
# Checkers and utils  #
#######################


def check_tick_interval(tick_interval: str) -> str:
    """
    Checks if argument is a Binance valid tick interval for candles.

    :param str tick_interval: A string, maybe, binance tick interval well formatted.
    :return: A string with the tick interval.
    """
    if tick_interval not in tick_interval_values:
        raise Exception(f"BinPan Error on tick_interval: {tick_interval} not in "
                        f"expected API intervals.\n{tick_interval_values}")
    return tick_interval


def open_from_milliseconds(ms: int, tick_interval: str) -> int:
    """
    Returns the open timestamp in milliseconds for a tick interval.

    :param ms: A timestamp in milliseconds.
    :param tick_interval: A tick interval like '1m', '1h', etc.
    :return: Open timestamp in milliseconds.
    """
    return KlineTimestamp(int(ms), tick_interval, "UTC").open


def next_open_by_milliseconds(ms: int, tick_interval: str) -> int:
    """
    Calculates the next open timestamp in milliseconds for a tick interval.

    :param ms: A timestamp in milliseconds.
    :param tick_interval: A tick interval like '1m', '1h', etc.
    :return: Next open timestamp in milliseconds.
    """
    return KlineTimestamp(int(ms), tick_interval, "UTC").next().open



def infer_frequency_and_set_index(data: pd.DataFrame, timestamp_column: str = 'Open timestamp',
                                  timezone: str = None) -> pd.DataFrame:
    """
    Infers the frequency of the DataFrame based on the 'Open timestamp' column and sets it as the DataFrame index.

    :param pd.DataFrame data: Input DataFrame with a column containing timestamps in milliseconds.
    :param str timestamp_column: Name of the column containing timestamps. Default is 'Open timestamp'.
    :param str timezone: The timezone to use. Default is None to infer from the dataframe.
    :return pd.DataFrame: DataFrame with the timestamp column set as the index and the frequency inferred.
    """
    timezone_from_data = data.index.name.split()[-1]
    if not timezone:
        timezone = timezone_from_data

    df = data.copy(deep=True)
    df = df.set_index(pd.to_datetime(df[timestamp_column], unit='ms').dt.tz_localize('UTC').dt.tz_convert(timezone))

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo datetime.")
    inferred_freq = pd.infer_freq(df.index)

    if inferred_freq:
        df = df.asfreq(inferred_freq)
    else:
        print("No se pudo inferir una frecuencia para el índice.")
    return df


def get_dataframe_time_index_ranges(data: pd.DataFrame, interval='30T') -> list[tuple]:
    """
    Divides a DataFrame into time ranges of a specified interval.

    :param pd.DataFrame data: A dataframe with a datetime index.
    :param interval: Interval expressed as a Pandas frequency string. Default is '30T'.
    :return list: A list of tuples containing the start and end time of each interval.
    """
    df_ = data.sort_index(ascending=False)
    start_time = df_.index.min()
    end_time = df_.index.max()
    time_ranges = []

    current_time = end_time

    while current_time > start_time:
        range_start = max(current_time - pd.Timedelta(interval), start_time)
        time_ranges.append((range_start, current_time))
        current_time -= pd.Timedelta(interval)

    return sorted(time_ranges)


def remove_initial_included_ranges(time_ranges, initial_minutes) -> list[tuple]:
    """
    Remove the time ranges that are completely included in the initial period.

    :param time_ranges: A list of tuples containing the start and end time of each interval.
    :param initial_minutes: A quantity of minutes that defines the initial period.
    :return: Filtered list of tuples.
    """
    if not time_ranges:
        return []

    initial_end_time = time_ranges[0][0] + pd.Timedelta(minutes=initial_minutes)

    modified_time_ranges = [(start_time, end_time) for start_time, end_time in time_ranges if end_time > initial_end_time]

    if not modified_time_ranges:
        return []

    final_ranges = [modified_time_ranges[0]]
    for start_time, end_time in modified_time_ranges[1:]:
        if start_time < final_ranges[-1][1]:
            continue
        final_ranges.append((start_time, end_time))

    return final_ranges
