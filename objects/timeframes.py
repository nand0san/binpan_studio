from datetime import datetime, timedelta
import pytz
from handlers.time_helper import parse_timestamp

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
    def __init__(self, value: any, timezone: str = "Europe/Madrid", tick_interval: str = "1m"):
        """
        Initializes the Timestamp object. The value can be a string, a datetime object, a timestamp in seconds (int),
        or other formats that can be interpreted as a datetime.

        The expected string format is "%Y-%m-%d %H:%M:%S", but other ISO 8601 formats can also be parsed.

        :param value: The value to be parsed and stored as a datetime object.
        :param timezone: The timezone for the timestamp, given as an IANA time zone string, e.g., "UTC", "Europe/Madrid".
                         Defaults to "Europe/Madrid".
        :param tick_interval: The tick interval for the timestamp, given as a string, e.g., "1m", "1h", "1d". Defaults to "1m".
        """
        assert tick_interval in tick_milliseconds, f"The tick interval is not recognized: {tick_interval}"

        try:
            self.timezone = pytz.timezone(timezone)
        except pytz.UnknownTimeZoneError:
            raise ValueError(f"The timezone is not recognized: {timezone}")

        self.dt = self.parse(value)
        self.ms = self.epoch()
        self.timestamp = self.ms
        self.tick_interval = tick_interval
        self.tick_interval_ms = tick_milliseconds[tick_interval]
        self.open = self.get_open()
        self.close = self.get_close()

    def parse(self, value) -> datetime:
        """
        Parses the value into a datetime object and adjusts it to the specified timezone.

        :param value: The value to be parsed into a datetime object.
        :return: The datetime object adjusted to the specified timezone.
        """
        # Assuming parse_timestamp is a previously defined function capable of handling different input formats
        return parse_timestamp(timestamp_str=value, timezone=self.timezone)

    def to_datetime(self) -> datetime:
        """
        Converts the Timestamp into a datetime object.

        :return: The datetime object.
        """
        return self.dt

    def to_string(self, format_: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Converts the Timestamp to a formatted string, including the timezone offset if present.

        :param format_: The output format of the string, without the timezone offset.
        :return: Formatted string according to the specified format, with the timezone offset appended if available.
        """
        # Formatea la fecha y hora según el formato especificado.
        formatted_str = self.dt.strftime(format_)

        # Obtiene el desplazamiento de la zona horaria como un timedelta.
        tz_offset = self.dt.utcoffset()

        if tz_offset is not None:
            # Convierte el desplazamiento de la zona horaria a horas y minutos.
            # Nota: tz_offset.total_seconds() devuelve el desplazamiento en segundos.
            # Se divide entre 3600 para obtener horas y se usa divmod para obtener horas y minutos.
            total_seconds = int(tz_offset.total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes = remainder // 60

            # Formatea el desplazamiento como +HH:MM o -HH:MM.
            tz_str = f"{'+' if total_seconds >= 0 else '-'}{hours:02}:{minutes:02}"
        else:
            # Si no hay información de la zona horaria, no añade nada.
            tz_str = ''

        # Combina la cadena formateada con el desplazamiento de la zona horaria.
        return f"{formatted_str} {tz_str}"

    def add_timedelta(self, delta: timedelta) -> 'Timestamp':
        """
        Adds a timedelta to the Timestamp.

        :param delta: The timedelta to add.
        :return: New Timestamp object with the added time.
        """
        new_dt = self.dt + delta
        return Timestamp(new_dt, self.timezone.zone)

    def subtract_timedelta(self, delta: timedelta) -> 'Timestamp':
        """
        Subtracts a timedelta from the Timestamp.

        :param delta: The timedelta to subtract.
        :return: New Timestamp object with the subtracted time.
        """
        new_dt = self.dt - delta
        return Timestamp(new_dt, self.timezone.zone)

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

    # Special methods for comparisons, representation, etc.
    def __repr__(self) -> str:
        return f"Timestamp({self.to_string()}, {self.timezone.zone})"

    def __eq__(self, other) -> bool:
        return self.dt == other.dt

    def __lt__(self, other) -> bool:
        """
        Checks if the Timestamp is less than another Timestamp. This enables comparisons like t1 < t2 or t1 < datetime.now().
        """
        return self.dt < other.dt



