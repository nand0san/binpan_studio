from datetime import datetime, timedelta
from time import time
import pytz
from typing import Union, Tuple

from objects.timestamps import Timestamp

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


class Timeframe:
    def __init__(self,
                 start: Union[str, int, datetime, Timestamp, None],
                 end: Union[str, int, datetime, Timestamp, None],
                 timezone_IANA: Union[str, None] = "Europe/Madrid",
                 tick_interval: Union[str, None] = "1m",
                 hours: int = None,
                 limit: int = None,
                 closed: bool = True
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
        :param closed: If True, the end of the Timeframe is included in the range. If False, the end is excluded because it is open.
                       Just for discrete timeframes. Default is True.
        """
        assert not (bool(limit) & bool(hours)), "Either 'hours' or 'limit' can be specified, but not both."

        self.tick_interval = tick_interval
        self.tick_interval_ms = tick_milliseconds[tick_interval] if tick_interval else None

        if self.tick_interval:
            assert self.tick_interval in tick_milliseconds, f"The tick interval is not recognized: {self.tick_interval}"
            self.is_discrete = True
        else:
            self.is_discrete = False

        if (not (bool(start) and bool(end))) and (limit or hours):
            start, end = self._get_start_end_from_hours_or_limit(start_time=start, end_time=end, limit=limit, hours=hours)

        # Ensure that the start and end are Timestamp objects.
        self.start = Timestamp(start, timezone_IANA=timezone_IANA)
        if not closed:
            self.end = Timestamp(end, timezone_IANA=timezone_IANA)
        else:
            now = int(time() * 1000)
            end_ms = min(Timestamp(end, timezone_IANA=timezone_IANA).ms,
                         Timestamp(now, timezone_IANA=timezone_IANA, tick_interval=self.tick_interval).get_prev_open())
            self.end = Timestamp(end_ms, timezone_IANA=timezone_IANA)
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
                                           end_time: Union[str, int, datetime, Timestamp] = None,
                                           closed: bool = True
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
        :param closed: If True, the end of the Timeframe is included in the range. If False, the end is excluded because it is open.
        :return: The start and end times of the Timeframe based on the number of hours or the limit.
        """
        assert bool(limit) ^ bool(hours), "Either 'hours' or 'limit' must be specified, but not both or none."

        if hours:
            # ceil division to ensure the limit is not less than the hours
            limit = -(-hours * 60 * 60 * 1000 // self.tick_interval_ms)

        now = Timestamp(int(time() * 1000), tick_interval=self.tick_interval)

        if closed:
            now_ts = now.get_prev_close()
        else:
            now_ts = now.epoch()

        if not end_time and not start_time:
            end_time = Timestamp(now_ts, tick_interval=self.tick_interval)
            start_time = end_time.subtract_timedelta(delta=limit * self.tick_interval_ms)

        elif not end_time and start_time:
            start_time = Timestamp(start_time, tick_interval=self.tick_interval)
            start_time.apply_open()
            end_time_ts = start_time.open + (limit * self.tick_interval_ms) - 1  # -1 ms para ajustar al close

            if closed:
                end_time = Timestamp(min(now_ts, end_time_ts), tick_interval=self.tick_interval)
            else:
                end_time = start_time.add_timedelta(delta=limit * self.tick_interval_ms)

        elif end_time and not start_time:
            end_time = Timestamp(now_ts, tick_interval=self.tick_interval)
            start_time = end_time.subtract_timedelta(delta=limit * self.tick_interval_ms)
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
