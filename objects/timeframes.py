from datetime import datetime, timedelta, timezone
from time import time
import pytz
from typing import Union, Tuple

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
    def __init__(self,
                 value: Union[str, int, datetime, 'Timestamp'],
                 timezone_IANA: str or None = "Europe/Madrid",
                 tick_interval: Union[str, None] = "1m"):
        """
        Initializes the Timestamp object. The value can be a string, a datetime object, an integer timestamp in milliseconds,
        or other formats that can be interpreted as a datetime.

        The expected string format is "%Y-%m-%d %H:%M:%S", but other ISO 8601 formats can also be parsed.

        :param value: The value to be parsed and stored as a datetime object. Allowed types are str, int, datetime, and Timestamp.
        :param timezone_IANA: The timezone for the timestamp, given as an IANA time zone string, e.g., "UTC", "Europe/Madrid".
                              Defaults to "Europe/Madrid". If not recognized or not defined, it defaults the value passed in the
                              timestamp if
                              it includes timezone information, or to UTC if it doesn't.
        :param tick_interval: The tick interval for the timestamp, given as a string, e.g., "1m", "1h", "1d". Defaults to "1m". If none is
                                specified, the timestamp open and close methods will raise an exception.

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
        if tick_interval:
            assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"
            self.tick_interval = tick_interval
            self.tick_interval_ms = tick_milliseconds[tick_interval]
        else:
            self.tick_interval = None
            self.tick_interval_ms = None

        if type(value) == 'Timestamp' or type(value) == Timestamp:
            self.dt = value.get_datetime()
        elif type(value) == str:
            self.dt = parse_timestamp(timestamp_str=value, timezone=timezone_IANA)
        elif type(value) == datetime:
            self.dt = value
        elif type(value) == int:
            # expected int in milliseconds
            assert value > 0, f"Timestamp value must be a positive integer, not {value}"
            self.dt = datetime.utcfromtimestamp(value / 1000).replace(tzinfo=pytz.utc)
        else:
            raise ValueError(f"Timestamp value must be other Timestamp, a string, a datetime object, or an integer timestamp in milliseconds, "
                             f"not {type(value)}")

        self.timezone = self.apply_new_timezone(timezone_=timezone_IANA)

        self.ms = self.epoch()
        self.timestamp = self.ms

        self.open = self.get_open() if self.tick_interval else None
        self.close = self.get_close() if self.tick_interval else None
        self.utc = self.dt.astimezone(pytz.utc)

        # asegura que el datetime de self.dt tenga una zona horaria con un offset compatible con self.timezone
        try:
            assert self.offset() == self.timezone_offset(), (f"El offset de la zona horaria de self.dt no es compatible con self.timezone. "
                                                             f"{self.offset()} != {self.timezone_offset()}")
        except AssertionError:
            self.apply_new_timezone(timezone_=timezone_IANA)

    def apply_new_timezone(self, timezone_: Union[datetime.tzinfo, str, None]) -> pytz or None:
        """
        Formats the Timestamp with the specified timezone in IANA format.
        """
        timezone_set = None
        if type(timezone_) == str:
            timezone_set = pytz.timezone(timezone_)
            self.dt = self.dt.astimezone(timezone_set)
        elif hasattr(timezone_, "zone"):
            timezone_set = timezone_
            self.dt = self.dt.astimezone(timezone_set)
        elif type(timezone_) == pytz.tzinfo.DstTzInfo or type(timezone_) == pytz.tzinfo.StaticTzInfo:
            timezone_set = timezone_
            self.dt = self.dt.astimezone(timezone_set)
        elif type(timezone_) == timezone:
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
        if self.tick_interval_ms is None:
            raise Exception(f"No tick interval defined for Timestamp: {self.to_string()}")
        return (self.ms // self.tick_interval_ms) * self.tick_interval_ms

    def get_close(self) -> int:
        """
        Returns the Timestamp at the end of the tick interval.

        :return: Milliseconds since the Unix epoch at the end of the tick interval.
        """
        if self.tick_interval_ms is None:
            raise Exception(f"No tick interval defined for Timestamp: {self.to_string()}")
        return self.get_open() + self.tick_interval_ms - 1

    def get_next_open(self) -> int:
        """
        Returns the Timestamp at the beginning of the next tick interval.

        :return: Milliseconds since the Unix epoch at the beginning of the next tick interval.
        """
        if self.tick_interval_ms is None:
            raise Exception(f"No tick interval defined for Timestamp: {self.to_string()}")
        return self.get_open() + self.tick_interval_ms

    def get_datetime(self) -> datetime:
        """
        Returns the datetime object of the Timestamp.

        :return: The datetime object.
        """
        return self.dt

    def update_interval(self, tick_interval: str) -> None:
        """
        Updates the tick interval and the corresponding milliseconds.

        :param tick_interval: The new tick interval.
        """
        if tick_interval:
            assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"
            self.tick_interval = tick_interval
            self.tick_interval_ms = tick_milliseconds[tick_interval]
            self.open = self.get_open()
            self.close = self.get_close()
        else:
            self.tick_interval = None
            self.tick_interval_ms = None
            self.open = None
            self.close = None

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

    def __add__(self, other):
        """
        Adds a timedelta or milliseconds to the Timestamp and returns a new Timestamp instance.

        :param other: The amount of time to add, either as a timedelta or as an integer representing milliseconds.
        :return: A new Timestamp instance with the added time.
        """
        if isinstance(other, timedelta):
            new_dt = self.dt + other
        elif isinstance(other, int):
            # Assume the integer represents milliseconds.
            delta = timedelta(milliseconds=other)
            new_dt = self.dt + delta
        else:
            raise TypeError("Addition only supports timedelta or integer (as milliseconds).")

        # Return a new Timestamp instance with the updated datetime.
        try:
            return Timestamp(new_dt, timezone_IANA=self.timezone.zone if self.timezone else None, tick_interval=self.tick_interval)
        except AttributeError:
            return Timestamp(new_dt, timezone_IANA=None, tick_interval=self.tick_interval)

    def __sub__(self, other):
        """
        Subtracts a timedelta or milliseconds from the Timestamp and returns a new Timestamp instance.

        :param other: The amount of time to subtract, either as a timedelta or as an integer representing milliseconds.
        :return: A new Timestamp instance with the subtracted time.
        """
        if isinstance(other, timedelta):
            new_dt = self.dt - other
        elif isinstance(other, int):
            # Assume the integer represents milliseconds.
            delta = timedelta(milliseconds=other)
            new_dt = self.dt - delta
        else:
            raise TypeError("Subtraction only supports timedelta or integer (as milliseconds).")

        # Return a new Timestamp instance with the updated datetime.
        try:
            return Timestamp(new_dt, timezone_IANA=self.timezone.zone if self.timezone else None, tick_interval=self.tick_interval)
        except AttributeError:
            return Timestamp(new_dt, timezone_IANA=None, tick_interval=self.tick_interval)

    def __len__(self) -> int:
        """
        Returns the number of milliseconds between the open and close of the tick interval.
        """
        return self.close - self.open


class Timeframe:
    def __init__(self,
                 start: Union[str, int, datetime, Timestamp, None],
                 end: Union[str, int, datetime, Timestamp, None],
                 timezone_IANA: Union[str, None] = "Europe/Madrid",
                 tick_interval: Union[str, None] = "1m",
                 hours: int = None,
                 limit: int = None
                 ):
        """
        Initializes the Timeframe object.

        :param start: The start of the Timeframe, given as a string, a datetime object, or an integer timestamp in milliseconds.
        :param end: The end of the Timeframe, given as a string, a datetime object, or an integer timestamp in milliseconds.
        :param timezone_IANA: The timezone for the Timeframe, given as an IANA time zone string, e.g., "UTC", "Europe/Madrid".
                              Defaults to "Europe/Madrid". If not recognized or not defined, it defaults the value passed in the
                              timestamp if it includes timezone information, or to UTC if it doesn't nad no timezone is specified.
        :param tick_interval: The tick interval for the Timeframe, given as a string, e.g., "1m", "1h", "1d". Defaults to "1m".
        :param hours: The number of hours of de range of Timeframe. If specified, it will be used to obtain start or end from the other.
        :param limit: The number of hours of the range of the Timeframe. If specified, it will be used to obtain start or end from the other.
        """
        assert not (bool(limit) & bool(hours)), "Either 'hours' or 'limit' can be specified, but not both."

        self.tick_interval = tick_interval
        self.tick_interval_ms = tick_milliseconds[tick_interval] if tick_interval else None

        if self.tick_interval:
            assert self.tick_interval in tick_milliseconds, f"The tick interval is not recognized: {self.tick_interval}"
            self.is_discrete = True
        else:
            self.is_discrete = False

        if not (bool(start) and bool(end)) and (limit or hours):
            start, end = self._get_start_end_from_hours_or_limit(start, end)

        # Ensure that the start and end are Timestamp objects.
        self.start = Timestamp(start, timezone_IANA=timezone_IANA)
        self.end = Timestamp(end, timezone_IANA=timezone_IANA)

        self.timezone = self.start.timezone
        self.ms = self.get_ms()

    def to_string(self) -> str:
        """
        Converts the Timeframe to a formatted string, including the timezone offset if present and defined.

        :return: Formatted string according to the specified format, with the timezone offset appended if available and defined.
        """
        start_str = self.start.to_string()
        end_str = self.end.to_string()
        return f"{start_str}, {end_str}, {self.timezone}, {self.tick_interval}"

    def __repr__(self) -> str:
        return f"Timeframe({self.start.to_string()}, {self.end.to_string()}, {self.timezone}, {self.tick_interval})"

    def __eq__(self, other) -> bool:
        return (self.start == other.start and self.end == other.end) and self.tick_interval == other.tick_interval

    def __contains__(self, item: Union[str, int, datetime, 'Timestamp']) -> bool:
        """
        Checks if a given timestamp is within the Timeframe.

        :param item: The timestamp to check, given as a string, an integer timestamp in milliseconds,
                     a datetime object, or a Timestamp object.
        :return: True if the timestamp is within the Timeframe, False otherwise.
        """
        # Primero, convertir 'item' a un objeto Timestamp si no lo es ya.
        if not isinstance(item, Timestamp):
            try:
                item_timestamp = Timestamp(item, timezone_IANA=self.timezone.zone)
            except AttributeError:
                item_timestamp = Timestamp(item, timezone_IANA=self.timezone)
        else:
            item_timestamp = item

        # Ahora, comprobar si el timestamp está dentro del intervalo.
        return self.start <= item_timestamp <= self.end

    def adjust_to_open(self, inplace: bool = False) -> 'Timeframe':
        """
        Adjusts the start and end to the beginning of their respective tick intervals.

        :param inplace: If True, modifies the Timeframe in place. If False, returns a new Timeframe object. Default is False.
        :return: If inplace is True, returns None. If inplace is False, returns a new Timeframe object.
        """
        new_start = Timestamp(self.start.get_open(), timezone_IANA=self.timezone, tick_interval=self.tick_interval)
        new_end = Timestamp(self.end.get_open(), timezone_IANA=self.timezone, tick_interval=self.tick_interval)
        if inplace:
            self.start = new_start
            self.end = new_end
            self.ms = self.get_ms()
            return self
        else:
            return Timeframe(new_start, new_end, self.timezone, self.tick_interval)

    def adjust_to_close(self, inplace: bool = False) -> 'Timeframe':
        """
        Adjusts the start and end to the end of their respective tick intervals.

        :param inplace: If True, modifies the Timeframe in place. If False, returns a new Timeframe object. Default is False.
        :return: If inplace is True, returns None. If inplace is False, returns a new Timeframe object.
        """
        new_start = Timestamp(self.start.get_close(), timezone_IANA=self.timezone, tick_interval=self.tick_interval)
        new_end = Timestamp(self.end.get_close(), timezone_IANA=self.timezone, tick_interval=self.tick_interval)
        if inplace:
            self.start = new_start
            self.end = new_end
            self.ms = self.get_ms()
            return self
        else:
            return Timeframe(new_start, new_end, self.timezone, self.tick_interval)

    def __add__(self, milliseconds: int) -> 'Timeframe':
        """Moves the timeframe forward by the specified number of milliseconds."""
        delta = timedelta(milliseconds=milliseconds)
        new_start = self.start + delta
        new_end = self.end + delta
        return Timeframe(new_start, new_end, self.timezone, self.tick_interval)

    def __sub__(self, milliseconds: int) -> 'Timeframe':
        """Moves the timeframe backward by the specified number of milliseconds."""
        delta = timedelta(milliseconds=milliseconds)
        new_start = self.start - delta
        new_end = self.end - delta
        return Timeframe(new_start, new_end, self.timezone, self.tick_interval)

    def __len__(self) -> int:
        """
        Returns the number of tick intervals within the timeframe if it is discrete, or the number of milliseconds otherwise.

        It is inclusive of the start and end times, so the length is the number of intervals including both times.
        """
        total_ms = self.end.ms - self.start.ms
        if self.is_discrete:
            return int(total_ms // self.tick_interval_ms) + 1
        else:
            print("Timeframe is not discrete. Returning total milliseconds.")
            return int(total_ms)

    def __gt__(self, other: 'Timeframe') -> bool:
        """
        Compares if this timeframe is longer than another.

        :param other: The other timeframe to compare.
        """
        return len(self) > len(other)

    def __lt__(self, other: 'Timeframe') -> bool:
        """
        Compares if this timeframe is shorter than another.

        :param other: The other timeframe to compare.
        """
        return len(self) < len(other)

    def __ge__(self, other: 'Timeframe') -> bool:
        """
        Compares if this timeframe is longer than or equal to another.

        :param other: The other timeframe to compare.
        """
        return len(self) >= len(other)

    def __le__(self, other: 'Timeframe') -> bool:
        """
        Compares if this timeframe is shorter than or equal to another.

        :param other: The other timeframe to compare.
        """
        return len(self) <= len(other)

    def get_ms(self) -> int:
        """Calculates the total milliseconds from start to end."""
        return int((self.end.to_datetime() - self.start.to_datetime()).total_seconds() * 1000)

    def update_tick_interval(self, tick_interval: str or None, inplace: bool = False) -> Union['Timeframe', None]:
        """
        Updates the tick interval and the corresponding milliseconds.

        :param tick_interval: The new tick interval.
        :param inplace: If True, modifies the Timeframe in place. If False, returns a new Timeframe object. Default is False.
        """
        if inplace:
            if tick_interval:
                assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"
                self.tick_interval_ms = tick_milliseconds[tick_interval]
            else:
                self.tick_interval_ms = None
            self.tick_interval = tick_interval
            self.is_discrete = True if tick_interval else False
            self.ms = self.get_ms()
        else:
            return Timeframe(self.start, self.end, self.timezone, tick_interval)

    def get_start(self) -> Timestamp:
        return self.start

    def get_end(self) -> Timestamp:
        return self.end

    def get_timezone(self) -> pytz or None:
        return self.timezone

    def get_tick_interval(self) -> str or None:
        return self.tick_interval

    def get_tick_interval_ms(self) -> int or None:
        return self.tick_interval_ms

    def __getitem__(self, key):
        """
        Allows indexed access to timestamps within the Timeframe based on tick intervals.

        :param key: The index (or slice) of the tick interval.
        :return: A Timestamp object representing the start of the tick interval for an index,
                 or a list of Timestamp objects for a slice.
        """
        if not self.is_discrete:
            raise TypeError("Indexing is only supported for discrete timeframes with a defined tick interval.")

        length = self.__len__()

        # Handling slices
        if isinstance(key, slice):
            start, stop, step = key.indices(length)  # Ajusta según la longitud del Timeframe.

            # Manejo especial para step negativo.
            if step < 0:
                # Asegúrate de que el rango es coherente con un paso negativo.
                if start >= stop:
                    return [self[i] for i in range(start, stop, step)]
                else:
                    return []  # Devuelve una lista vacía si el rango y el paso están en conflicto.
            else:
                return [self[i] for i in range(start, stop, step)]

        # Handling single index
        elif isinstance(key, int):
            if key < 0:  # Adjust negative index
                key += length
            if key < 0 or key >= length:
                raise IndexError("Timeframe index out of range.")
            tick_start_time = self.start.to_datetime() + timedelta(milliseconds=self.tick_interval_ms * key)
            return Timestamp(tick_start_time, timezone_IANA=self.timezone.zone, tick_interval=self.tick_interval)

        else:
            raise TypeError("Invalid index type. Must be an integer or slice.")

    def __iter__(self):
        """
        Allows iteration over the Timeframe, returning a Timestamp for each tick interval's open time.

        :return: An iterator over each tick interval within the Timeframe.
        """
        if not self.is_discrete:
            raise ValueError("Iteration is only supported for discrete timeframes with a defined tick interval.")
        return TimeframeIterator(self)

    def get_hours(self) -> float:
        """
        Returns the number of hours of the range of the Timeframe.
        """
        if self.is_discrete:
            return len(self) * self.tick_interval_ms / (60 * 60 * 1000)
        else:
            return (self.end.to_datetime() - self.start.to_datetime()).total_seconds() / 3600

    def get_limit(self) -> int:
        """
        Returns the number of hours of the range of the Timeframe.
        """
        return len(self)

    def _get_start_end_from_hours_or_limit(self,
                                           hours: Union[float, None],
                                           limit: Union[int, None],
                                           start_time: Union[str, int, datetime, Timestamp] = None,
                                           end_time: Union[str, int, datetime, Timestamp] = None
                                           ) -> Tuple[Timestamp, Timestamp]:
        """
        Returns the number of hours of the range of the Timeframe based on the limit.

        If the limit is specified, it will be used to obtain start or end from the other.

        If the hours is specified, it will be used to obtain start or end from the other.

        If both are specified, it will raise an exception.

        If none is specified, it will raise an exception.

        :param hours: The number of hours of the range of the Timeframe. If specified, it will be used to obtain start or end from the other.
                      It is prioritized over the limit.
        :param limit: The number of hours of the range of the Timeframe. If specified, it will be used to obtain start or end from the other.
                      It is used only if hours is not specified.
        :param start_time: The start of the Timeframe, given as a string, a datetime object, or an integer timestamp in milliseconds,
                           even a Timestamp object.
        :param end_time: The end of the Timeframe, given as a string, a datetime object, or an integer timestamp in milliseconds,
                         even a Timestamp object.
        :return: The start and end times of the Timeframe based on the number of hours or the limit.
        """
        assert bool(limit) ^ bool(hours), "Either 'hours' or 'limit' must be specified, but not both or none."

        if hours:
            # ceil division to ensure the limit is not less than the hours
            limit = -(-hours * 60 * 60 * 1000 // self.tick_interval_ms)

        if not end_time and not start_time:
            now = int(1000 * time())
            end_time = Timestamp(value=now, tick_interval=self.tick_interval)
            start_time = end_time.subtract_timedelta(delta=limit * 60 * 60 * 1000)

        elif not end_time and start_time:
            start_time = Timestamp(start_time, tick_interval=self.tick_interval)
            end_time = start_time.add_timedelta(delta=limit * 60 * 60 * 1000)

        elif end_time and not start_time:
            end_time = Timestamp(end_time, tick_interval=self.tick_interval)
            start_time = end_time.subtract_timedelta(delta=limit * 60 * 60 * 1000)
        else:
            raise NotImplementedError(f"This case is not implemented yet. start_time: {start_time}, end_time: {end_time}, limit: {limit}, hours: {hours}")
        return start_time, end_time


class TimeframeIterator:
    """
    Iterator class for Timeframe to iterate through each tick interval.
    """

    def __init__(self, timeframe):
        # ensures it is a discrete timeframe
        assert timeframe.is_discrete, "Timeframe not iterable, it's not discrete."
        self._timeframe = timeframe
        self._current = self._timeframe.start.to_datetime()
        # print(f"Current type and timezone in _current: {type(self._timeframe.timezone)}, {self._timeframe.timezone}")

    def __iter__(self):
        return self

    def __next__(self):
        if self._current > self._timeframe.end.to_datetime():
            raise StopIteration

        current_timestamp = Timestamp(self._current, timezone_IANA=self._timeframe.timezone, tick_interval=self._timeframe.tick_interval)

        # Move to the next open based on the tick interval.
        self._current += timedelta(milliseconds=self._timeframe.tick_interval_ms)
        return current_timestamp
