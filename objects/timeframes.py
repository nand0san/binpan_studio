from datetime import datetime, timedelta, timezone
import pytz
from typing import Union

tick_milliseconds = {
    '1m': 60 * 1000,
    '3m': 3 * 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '2h': 2 * 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '6h': 6 * 60 * 60 * 1000,
    '8h': 8 * 60 * 60 * 1000,
    '12h': 12 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '3d': 3 * 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000,
    '1M': 30 * 24 * 60 * 60 * 1000,  # Approximation for one month
}


def parse_timestamp(timestamp_str: str, timezone: Union[str, pytz] = "Europe/Madrid") -> datetime:
    """
    Parses a timestamp in ISO 8601 format or in one of the specified custom formats. Accepted formats are:

    .. code-block:: python

        formats = [
            "%Y-%m-%d %H:%M:%S%z",  # Formatos con información de zona horaria
            "%Y/%m/%d %H:%M:%S%z",
            "%d-%m-%Y %H:%M:%S%z",
            "%d/%m/%Y %H:%M:%S%z",
            "%Y%m%d%H%M%S%z",
            "%Y-%m-%d %H:%M:%S",  # Formatos sin información de zona horaria
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%Y%m%d%H%M%S",
            "%Y-%m-%d",  # Formatos de solo fecha no necesitan zona horaria
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y%m%d",
        ]

    If the timestamp string don't include time zone information, it is assumed to be in UTC.
    If the timestamp string includes time zone information, the resulting datetime object will be timezone-aware.
    If timezone parameter is specified, and the timestamp string includes time zone information with a different offset,
    it adjusts the datetime object to utc and then to the specified timezone. If the offsets are the same, it simply makes the datetime
    object aware of the specified timezone without changing the time.

    :param timestamp_str: Timestamp in ISO 8601 format or in one of the specified custom formats.
    :param timezone: Time zone in IANA format (e.g., "Europe/Madrid"). If specified, the datetime object will be adjusted
                     to this timezone. If not specified, and the timestamp includes time zone information,
                     the datetime object will remain in its original timezone. If no time zone information is included,
                     the timestamp is assumed to be in UTC. Default is "Europe/Madrid".
    :return: Datetime object with the date and time specified in the timestamp, adjusted to the specified timezone if provided. Example:

        .. code-block:: python

            dt = parse_timestamp("2024-01-27 03:14:00", None)
            print(dt, dt.tzinfo)
            2024-01-27 03:14:00+00:00 UTC

            dt = parse_timestamp("2024-01-27 03:14:00+01:00", None)
            print(dt, dt.tzinfo)
            2024-01-27 02:14:00+00:00 UTC

            dt = parse_timestamp("01-01-2024 3:14:00", "Europe/Madrid")
            print(dt, dt.tzinfo)
            2024-01-01 04:14:00+01:00 Europe/Madrid

            dt = parse_timestamp("2024-01-27 03:14:00+01:00", "Europe/Madrid")
            print(dt, dt.tzinfo)
            2024-01-27 03:14:00+01:00 Europe/Madrid

            dt = parse_timestamp("2024-01-27 03:14:00+01:00", pytz.timezone("Europe/Madrid"))
            print(dt, dt.tzinfo)
            2024-01-27 03:14:00+01:00 Europe/Madrid

            dt = parse_timestamp("01-01-2024", None)
            print(dt, dt.tzinfo)
            2024-01-01 00:00:00+00:00 UTC

    """
    formats = [
        "%Y-%m-%d %H:%M:%S%z",  # Formatos con información de zona horaria
        "%Y/%m/%d %H:%M:%S%z",
        "%d-%m-%Y %H:%M:%S%z",
        "%d/%m/%Y %H:%M:%S%z",
        "%Y%m%d%H%M%S%z",
        "%Y-%m-%d %H:%M:%S",  # Formatos sin información de zona horaria
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y%m%d%H%M%S",
        "%Y-%m-%d",  # Formatos de solo fecha no necesitan zona horaria
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
    elif timezone and hasattr(timezone, 'localize') or hasattr(timezone, 'utcoffset'):
        pass  # It's a pytz timezone object or similar; no change needed
    else:
        raise TypeError(f"Timezone must be a string or a pytz timezone object, not {type(timezone)}")

    if timezone:
        dt = dt.astimezone(timezone)
    return dt


class Timestamp:
    def __init__(self, value: Union[str, int, datetime], timezone_IANA: str or None = "Europe/Madrid", tick_interval: str = "1m"):
        """
        Initializes the Timestamp object. The value can be a string, a datetime object, an integer timestamp in milliseconds,
        or other formats that can be interpreted as a datetime.

        The expected string format is "%Y-%m-%d %H:%M:%S", but other ISO 8601 formats can also be parsed.

        :param value: The value to be parsed and stored as a datetime object.
        :param timezone_IANA: The timezone for the timestamp, given as an IANA time zone string, e.g., "UTC", "Europe/Madrid".
                              Defaults to "Europe/Madrid". If not recognized or not defined, it defaults the value passed in the
                              timestamp if
                              it includes timezone information, or to UTC if it doesn't.
        :param tick_interval: The tick interval for the timestamp, given as a string, e.g., "1m", "1h", "1d". Defaults to "1m".

        Example:

            .. code-block:: python

                >>> dt = Timestamp("2024-01-27 03:14:00+01:00", None)
                Timestamp(2024-01-27 02:14:00+00:00, UTC)

                >>> dt = Timestamp("2024-01-27 03:14:15", None)
                Timestamp(2024-01-27 03:14:15+00:00, UTC)

                >>> dt = Timestamp(1706325255000, None)
                Timestamp(2024-01-27 03:14:15+00:00, UTC)

                >>> dt = Timestamp(1706325255000, "Europe/Madrid")
                Timestamp(2024-01-27 04:14:15+01:00, Europe/Madrid)
                >>> dt.to_datetime()
                datetime.datetime(2024, 1, 27, 4, 14, 15, tzinfo=<DstTzInfo 'Europe/Madrid' CET+1:00:00 STD>)
                >>> dt.to_string()
                '2024-01-27 04:14:15+01:00'


        """
        assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"

        if type(value) == str:
            self.dt = parse_timestamp(timestamp_str=value, timezone=timezone_IANA)
        elif type(value) == datetime:
            self.dt = value
        elif type(value) == int:
            # expected int in milliseconds
            assert value > 0, f"Timestamp value must be a positive integer, not {value}"
            self.dt = datetime.utcfromtimestamp(value / 1000).replace(tzinfo=pytz.utc)
        else:
            raise ValueError(f"Timestamp value must be a string, a datetime object, or an integer timestamp in milliseconds, "
                             f"not {type(value)}")

        self.timezone = self.apply_new_timezone(timezone_=timezone_IANA)

        self.tick_interval = tick_interval
        self.tick_interval_ms = tick_milliseconds[tick_interval]

        self.ms = self.epoch()
        self.timestamp = self.ms

        self.open = self.get_open()
        self.close = self.get_close()
        self.utc = self.dt.astimezone(pytz.utc)

        # asegura que el datetime de self.dt tenga una zona horaria con un offset compatible con self.timezone
        assert self.offset() == self.timezone_offset(), f"El offset de la zona horaria de self.dt no es compatible con self.timezone. {self.offset()} != {self.timezone_offset()}"

    def apply_new_timezone(self, timezone_: Union[datetime.tzinfo, str, None]) -> pytz or None:
        """
        Formats the Timestamp with the specified timezone in IANA format.
        """
        timezone_set = None
        if type(timezone_) == str:
            timezone_set = pytz.timezone(timezone_)
            self.dt = self.dt.astimezone(timezone_set)
        elif type(timezone_) == pytz.tzinfo.DstTzInfo:
            timezone_set = timezone_
            self.dt = self.dt.astimezone(timezone_set)
        # caso de timezone nativa de datetime y no se especifica timezone
        elif self.dt.tzinfo and not timezone_:
            # Inherits the timezone from the datetime object
            timezone_set = self.dt.tzinfo
        return timezone_set

    def to_datetime(self) -> datetime:
        """
        Converts the Timestamp into a datetime object.

        :return: The datetime object.
        """
        return self.dt

    def to_string(self, format_: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Converts the Timestamp to a formatted string, including the timezone offset if present and defined.

        :param format_: The output format of the string, without the timezone offset.
        :return: Formatted string according to the specified format, with the timezone offset appended if available and defined.
        """
        # Formatea la fecha y hora según el formato especificado.
        formatted_str = self.dt.strftime(format_)

        # Verifica si la zona horaria está definida.
        if self.timezone is not None and self.dt.utcoffset() is not None:
            # Calculate the timezone offset in hours and minutes.
            total_seconds = int(self.dt.utcoffset().total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60
            tz_str = f"{'+' if total_seconds >= 0 else '-'}{hours:02}:{minutes:02}"
        else:
            tz_str = ''

        return f"{formatted_str}{tz_str}".strip()

    def __repr__(self) -> str:
        # timezone_str = "None" if self.timezone is None else self.timezone
        # return f"Timestamp({self.to_string()}, {timezone_str})"
        string = self.to_string()
        return f"Timestamp({string}, {self.timezone})"

    def add_timedelta(self, delta: timedelta or int) -> 'Timestamp':
        """
        Adds a timedelta to the Timestamp.

        :param delta: The timedelta to add. If an integer is given, it is interpreted as milliseconds.
        :return: New Timestamp object with the added time.
        """
        if isinstance(delta, int):
            delta = timedelta(milliseconds=delta)
        new_dt = self.dt + delta
        return Timestamp(new_dt, self.timezone.zone if self.timezone else None, tick_interval=self.tick_interval)

    def subtract_timedelta(self, delta: timedelta or int) -> 'Timestamp':
        """
        Subtracts a timedelta from the Timestamp.

        :param delta: The timedelta to subtract. If an integer is given, it is interpreted as milliseconds.
        :return: New Timestamp object with the subtracted time.
        """
        if isinstance(delta, int):
            delta = timedelta(milliseconds=delta)
        new_dt = self.dt - delta
        return Timestamp(new_dt, self.timezone.zone if self.timezone else None, tick_interval=self.tick_interval)

    def epoch(self, milliseconds: bool = True) -> int:
        """
        Gets the timestamp value as seconds or milliseconds since the Unix epoch.

        :param milliseconds: If True, returns milliseconds.
        :return: Seconds or milliseconds since the Unix epoch.
        """
        # epoch = self.dt.replace(tzinfo=pytz.utc).timestamp()
        if not self.dt.tzinfo:
            dt = self.dt.replace(tzinfo=pytz.utc)
        else:
            dt = self.dt
        epoch = dt.timestamp()
        return int(epoch * 1000) if milliseconds else int(epoch)

    def get_open(self) -> int:
        """
        Returns the Timestamp at the beginning of the tick interval.

        :return: Milliseconds since the Unix epoch at the beginning of the tick interval.
        """
        return (self.ms // self.tick_interval_ms) * self.tick_interval_ms

    def get_close(self) -> int:
        """
        Returns the Timestamp at the end of the tick interval.

        :return: Milliseconds since the Unix epoch at the end of the tick interval.
        """
        return self.get_open() + self.tick_interval_ms - 1

    def get_next_open(self) -> int:
        """
        Returns the Timestamp at the beginning of the next tick interval.

        :return: Milliseconds since the Unix epoch at the beginning of the next tick interval.
        """
        return self.get_open() + self.tick_interval_ms

    def update_interval(self, tick_interval: str) -> None:
        """
        Updates the tick interval and the corresponding milliseconds.

        :param tick_interval: The new tick interval.
        """
        assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"
        self.tick_interval = tick_interval
        self.tick_interval_ms = tick_milliseconds[tick_interval]
        self.open = self.get_open()
        self.close = self.get_close()

    def offset(self) -> timedelta:
        """
        Returns the offset of the Timestamp from UTC.
        """
        return self.dt.utcoffset()

    def timezone_offset(self) -> timedelta:
        """
        Returns the offset of self.timezone from UTC. Timezone is assumed to be a pytz timezone object, a datetime.timezone object, or None.
        """
        # print(self.timezone, type(self.timezone))
        if self.timezone is None:
            return timedelta(seconds=0)
        elif hasattr(self.timezone, 'localize'):
            current_utc_dt = datetime.now(pytz.utc)
            madrid_dt = current_utc_dt.astimezone(self.timezone)
            offset = madrid_dt.utcoffset()
            return offset
        elif type(self.timezone) == timezone:
            return self.timezone.utcoffset(datetime.now())
        else:
            raise TypeError(f"El objeto self.timezone no es reconocido: {type(self.timezone)}")

    def __eq__(self, other) -> bool:
        # Asegúrate de que 'other' sea un objeto datetime o Timestamp para la comparación.
        other_dt = self._convert_to_comparable_datetime(other)
        return self.dt == other_dt

    def __lt__(self, other) -> bool:
        other_dt = self._convert_to_comparable_datetime(other)
        return self.dt < other_dt

    def __le__(self, other) -> bool:
        other_dt = self._convert_to_comparable_datetime(other)
        return self.dt <= other_dt

    def __gt__(self, other) -> bool:
        other_dt = self._convert_to_comparable_datetime(other)
        return self.dt > other_dt

    def __ge__(self, other) -> bool:
        other_dt = self._convert_to_comparable_datetime(other)
        return self.dt >= other_dt

    def _convert_to_comparable_datetime(self, other) -> datetime:
        """Convierte 'other' a un objeto datetime comparable con 'self.dt', considerando las zonas horarias."""
        if isinstance(other, Timestamp):
            other_dt = other.dt
        else:
            other_dt = Timestamp(other, timezone_IANA=self.timezone.zone).dt
            if not isinstance(other, datetime):
                raise TypeError(f"Cannot compare Timestamp with object of type {type(other)}")
        return other_dt


class Timeframe:
    def __init__(self,
                 start: Union[str, int, datetime],
                 end: Union[str, int, datetime],
                 timezone_IANA: str or None = "Europe/Madrid",
                 tick_interval: str = "1m"
                 ):
        """
        Initializes the Timeframe object.

        :param start: The start of the Timeframe, given as a string, a datetime object, or an integer timestamp in milliseconds.
        :param end: The end of the Timeframe, given as a string, a datetime object, or an integer timestamp in milliseconds.
        :param timezone_IANA: The timezone for the Timeframe, given as an IANA time zone string, e.g., "UTC", "Europe/Madrid".
                              Defaults to "Europe/Madrid". If not recognized or not defined, it defaults the value passed in the
                              timestamp if it includes timezone information, or to UTC if it doesn't nad no timezone is specified.
        :param tick_interval: The tick interval for the Timeframe, given as a string, e.g., "1m", "1h", "1d". Defaults to "1m".
        """
        assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"
        self.start = Timestamp(start, timezone_IANA=timezone_IANA)
        self.end = Timestamp(end, timezone_IANA=timezone_IANA)
        self.timezone = self.start.timezone

    def to_string(self) -> str:
        """
        Converts the Timeframe to a formatted string, including the timezone offset if present and defined.

        :return: Formatted string according to the specified format, with the timezone offset appended if available and defined.
        """
        start_str = self.start.to_string()
        end_str = self.end.to_string()
        return f"Timeframe({start_str}, {end_str}, {self.timezone.zone})"

    def __repr__(self) -> str:
        return self.to_string()

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.end == other.end

    def __contains__(self, item: Union[str, int, datetime, 'Timestamp']) -> bool:
        """
        Checks if a given timestamp is within the Timeframe.

        :param item: The timestamp to check, given as a string, an integer timestamp in milliseconds,
                     a datetime object, or a Timestamp object.
        :return: True if the timestamp is within the Timeframe, False otherwise.
        """
        # Primero, convertir 'item' a un objeto Timestamp si no lo es ya.
        if not isinstance(item, Timestamp):
            item_timestamp = Timestamp(item, timezone_IANA=self.timezone.zone)
        else:
            item_timestamp = item

        # Ahora, comprobar si el timestamp está dentro del intervalo.
        return self.start <= item_timestamp <= self.end


