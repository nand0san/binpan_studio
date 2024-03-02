from datetime import datetime, timedelta
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


class Timestamp:
    def __init__(self, value: any, timezone: str or None = "Europe/Madrid", tick_interval: str = "1m"):
        """
        Initializes the Timestamp object. The value can be a string, a datetime object, a timestamp in seconds (int),
        or other formats that can be interpreted as a datetime.

        The expected string format is "%Y-%m-%d %H:%M:%S", but other ISO 8601 formats can also be parsed.

        :param value: The value to be parsed and stored as a datetime object.
        :param timezone: The timezone for the timestamp, given as an IANA time zone string, e.g., "UTC", "Europe/Madrid".
                         Defaults to "Europe/Madrid". If not recognized, it defaults to None, that is local time.
        :param tick_interval: The tick interval for the timestamp, given as a string, e.g., "1m", "1h", "1d". Defaults to "1m".
        """
        assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"

        self.dt = parse_timestamp(timestamp_str=value, timezone=timezone)
        # extract time zone from the datetime object
        self.timezone = None
        if self.dt.tzinfo is not None:
            self.timezone = self.dt.tzinfo

        self.ms = self.epoch()
        self.timestamp = self.ms
        self.tick_interval = tick_interval
        self.tick_interval_ms = tick_milliseconds[tick_interval]
        self.open = self.get_open()
        self.close = self.get_close()
        self.utc = self.dt.astimezone(pytz.utc)

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
            # Convierte el desplazamiento de la zona horaria a horas y minutos.
            total_seconds = int(self.dt.utcoffset().total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60

            # Formatea el desplazamiento como +HH:MM o -HH:MM.
            tz_str = f"{'+' if total_seconds >= 0 else '-'}{hours:02}:{minutes:02}"
        else:
            # Si no hay información de la zona horaria, no añade nada.
            tz_str = ''

        # Combina la cadena formateada con el desplazamiento de la zona horaria, si está disponible.
        return f"{formatted_str}{tz_str if tz_str else ''}".strip()

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
        epoch = self.dt.replace(tzinfo=pytz.utc).timestamp()
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

    def __repr__(self) -> str:
        timezone_str = "None" if self.timezone is None else self.timezone.zone
        return f"Timestamp({self.to_string()}, {timezone_str})"

    def __eq__(self, other) -> bool:
        if self.dt.tzinfo is None and other.dt.tzinfo is not None:
            # Si self.dt es naive y other.dt es aware, convierte other.dt a naive asumiendo UTC
            other_dt = other.dt.replace(tzinfo=None)
            return self.dt == other_dt
        elif self.dt.tzinfo is not None and other.dt.tzinfo is None:
            # Si self.dt es aware y other.dt es naive, convierte self.dt a naive asumiendo UTC
            self_dt = self.dt.replace(tzinfo=None)
            return self_dt == other.dt
        else:
            # Si ambos son aware o ambos son naive, compara directamente
            return self.dt == other.dt
    #
    # def __lt__(self, other) -> bool:
    #     """
    #     Checks if the Timestamp is less than another Timestamp.
    #     """
    #     if self.dt.tzinfo is not None and other.dt.tzinfo is None:
    #         # Si self.dt es aware y other.dt es naive, convierte self.dt a naive UTC para comparar
    #         self_dt = self.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self_dt < other.dt
    #     elif self.dt.tzinfo is None and other.dt.tzinfo is not None:
    #         # Si self.dt es naive y other.dt es aware, convierte other.dt a naive UTC para comparar
    #         other_dt = other.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self.dt < other_dt
    #     elif self.dt.tzinfo is not None and other.dt.tzinfo is not None:
    #         # Si ambos son aware, convierte ambos a UTC para una comparación justa
    #         self_dt = self.dt.astimezone(pytz.utc)
    #         other_dt = other.dt.astimezone(pytz.utc)
    #         return self_dt < other_dt
    #     else:
    #         # Si ambos son naive, compara directamente
    #         return self.dt < other.dt
    #
    # def __gt__(self, other) -> bool:
    #     """
    #     Checks if the Timestamp is greater than another Timestamp.
    #     """
    #     if self.dt.tzinfo is not None and other.dt.tzinfo is None:
    #         # Si self.dt es aware y other.dt es naive, convierte self.dt a naive UTC para comparar
    #         self_dt = self.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self_dt > other.dt
    #     elif self.dt.tzinfo is None and other.dt.tzinfo is not None:
    #         # Si self.dt es naive y other.dt es aware, convierte other.dt a naive UTC para comparar
    #         other_dt = other.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self.dt > other_dt
    #     elif self.dt.tzinfo is not None and other.dt.tzinfo is not None:
    #         # Si ambos son aware, convierte ambos a UTC para una comparación justa
    #         self_dt = self.dt.astimezone(pytz.utc)
    #         other_dt = other.dt.astimezone(pytz.utc)
    #         return self_dt > other_dt
    #     else:
    #         # Si ambos son naive, compara directamente
    #         return self.dt > other.dt
    #
    # def __le__(self, other) -> bool:
    #     """
    #     Checks if the Timestamp is less or equal than another Timestamp.
    #     """
    #     if self.dt.tzinfo is not None and other.dt.tzinfo is None:
    #         # Si self.dt es aware y other.dt es naive, convierte self.dt a naive UTC para comparar
    #         self_dt = self.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self_dt <= other.dt
    #     elif self.dt.tzinfo is None and other.dt.tzinfo is not None:
    #         # Si self.dt es naive y other.dt es aware, convierte other.dt a naive UTC para comparar
    #         other_dt = other.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self.dt <= other_dt
    #     elif self.dt.tzinfo is not None and other.dt.tzinfo is not None:
    #         # Si ambos son aware, convierte ambos a UTC para una comparación justa
    #         self_dt = self.dt.astimezone(pytz.utc)
    #         other_dt = other.dt.astimezone(pytz.utc)
    #         return self_dt <= other_dt
    #     else:
    #         # Si ambos son naive, compara directamente
    #         return self.dt <= other.dt
    #
    # def __ge__(self, other) -> bool:
    #     """
    #     Checks if the Timestamp is greater or equal than another Timestamp.
    #     """
    #     if self.dt.tzinfo is not None and other.dt.tzinfo is None:
    #         # Si self.dt es aware y other.dt es naive, convierte self.dt a naive UTC para comparar
    #         self_dt = self.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self_dt >= other.dt
    #     elif self.dt.tzinfo is None and other.dt.tzinfo is not None:
    #         # Si self.dt es naive y other.dt es aware, convierte other.dt a naive UTC para comparar
    #         other_dt = other.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self.dt >= other_dt
    #     elif self.dt.tzinfo is not None and other.dt.tzinfo is not None:
    #         # Si ambos son aware, convierte ambos a UTC para una comparación justa
    #         self_dt = self.dt.astimezone(pytz.utc)
    #         other_dt = other.dt.astimezone(pytz.utc)
    #         return self_dt >= other_dt
    #     else:
    #         # Si ambos son naive, compara directamente
    #         return self.dt >= other.dt
    #
    # def __ne__(self, other) -> bool:
    #     """
    #     Checks if the Timestamp is not equal to another Timestamp, considering timezone differences.
    #     """
    #     if self.dt.tzinfo is not None and other.dt.tzinfo is None:
    #         # Si self.dt es aware y other.dt es naive, convierte self.dt a naive UTC para comparar
    #         self_dt = self.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self_dt != other.dt
    #     elif self.dt.tzinfo is None and other.dt.tzinfo is not None:
    #         # Si self.dt es naive y other.dt es aware, convierte other.dt a naive UTC para comparar
    #         other_dt = other.dt.astimezone(pytz.utc).replace(tzinfo=None)
    #         return self.dt != other_dt
    #     elif self.dt.tzinfo is not None and other.dt.tzinfo is not None:
    #         # Si ambos son aware, convierte ambos a UTC para una comparación justa
    #         self_dt = self.dt.astimezone(pytz.utc)
    #         other_dt = other.dt.astimezone(pytz.utc)
    #         return self_dt != other_dt
    #     else:
    #         # Si ambos son naive, compara directamente
    #         return self.dt != other.dt


def parse_timestamp(timestamp_str: str, timezone: Union[str, pytz] = "Europe/Madrid") -> datetime:
    """
    Parses a timestamp in ISO 8601 format or in one of the specified custom formats. Accepted formats are:

    - "YYYY-MM-DD HH:MM:SS"
    - "YYYY/MM/DD HH:MM:SS"
    - "YYYY-MM-DD"
    - "YYYY/MM/DD"
    - "DD-MM-YYYY HH:MM:SS"
    - "DD/MM/YYYY HH:MM:SS"
    - "DD-MM-YYYY"
    - "DD/MM/YYYY"
    - "YYYYMMDDHHMMSS"
    - "YYYYMMDD"

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

    # noinspection PyUnresolvedReferences
    if timezone is None:
        pass
    elif isinstance(timezone, str):
        timezone = pytz.timezone(timezone)
    elif timezone and hasattr(timezone, 'localize') or hasattr(timezone, 'utcoffset'):
        # It's a pytz timezone object or similar; no change needed
        pass
    else:
        raise TypeError(f"Timezone must be a string or a pytz timezone object, not {type(timezone)}")

    if timezone:
        dt = dt.astimezone(timezone)
    return dt

