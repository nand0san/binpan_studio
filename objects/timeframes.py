"""

Timeframe class: represents a discrete time range of klines.

Uses KlineTimestamp from kline-timestamp library for candle boundary calculations.

"""

from datetime import datetime, timedelta
import pandas as pd
from time import time
import pytz

from kline_timestamp import KlineTimestamp
from handlers.time_helper import parse_timestamp, tick_seconds

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


def _to_ms(value: str | int | datetime | KlineTimestamp, timezone: str | None = "UTC") -> int:
    """
    Converts various types to milliseconds from epoch.
    """
    if isinstance(value, KlineTimestamp):
        return value.open
    elif isinstance(value, int):
        return value
    elif isinstance(value, str):
        dt = parse_timestamp(value, timezone=timezone)
        return int(dt.timestamp() * 1000)
    elif isinstance(value, datetime):
        if value.tzinfo is None:
            value = pytz.utc.localize(value)
        return int(value.timestamp() * 1000)
    else:
        raise TypeError(f"Cannot convert {type(value)} to milliseconds")


class Timeframe:
    def __init__(self,
                 start: str | int | datetime | KlineTimestamp | None,
                 end: str | int | datetime | KlineTimestamp | None,
                 timezone_IANA: str | None = "UTC",
                 tick_interval: str | None = "1m",
                 hours: int = None,
                 limit: int = None,
                 closed: bool = True
                 ):
        """
        Initializes the Timeframe object.

        :param start: The start of the Timeframe (string, int ms, datetime, or KlineTimestamp).
        :param end: The end of the Timeframe (string, int ms, datetime, or KlineTimestamp).
        :param timezone_IANA: The timezone in IANA format. Defaults to "UTC".
        :param tick_interval: The tick interval (e.g., "1m", "1h", "1d"). Defaults to "1m".
        :param hours: Number of hours for the range. Mutually exclusive with limit.
        :param limit: Number of candles for the range. Mutually exclusive with hours.
        :param closed: If True, the end is capped to the last closed candle. Default is True.
        """
        assert not (bool(limit) & bool(hours)), "Either 'hours' or 'limit' can be specified, but not both."

        self.tick_interval = tick_interval
        self.tick_interval_ms = tick_seconds[tick_interval] * 1000 if tick_interval else None
        self.timezone_IANA = timezone_IANA or "UTC"
        self.is_discrete = bool(tick_interval)

        if (not (bool(start) and bool(end))) and (limit or hours):
            start, end = self._get_start_end_from_hours_or_limit(
                start_time=start, end_time=end, limit=limit, hours=hours)

        assert start is not None and end is not None, "Not enough information to create the Timeframe."

        start_ms = _to_ms(start, self.timezone_IANA)
        end_ms = _to_ms(end, self.timezone_IANA)

        if closed and self.is_discrete:
            now = int(time() * 1000)
            now_prev_open = KlineTimestamp(now, self.tick_interval, self.timezone_IANA).prev().open
            end_ms = min(end_ms, now_prev_open)

        self.start = KlineTimestamp(start_ms, self.tick_interval, self.timezone_IANA) if self.is_discrete else start_ms
        self.end = KlineTimestamp(end_ms, self.tick_interval, self.timezone_IANA) if self.is_discrete else end_ms

        self.start_ms = self.start.open if self.is_discrete else start_ms
        self.end_ms = self.end.open if self.is_discrete else end_ms
        self.ms = self.end_ms - self.start_ms

    def to_string(self) -> str:
        """
        Converts the Timeframe to a formatted string.
        """
        if self.is_discrete:
            return f"{self.start.to_datetime()}, {self.end.to_datetime()}, {self.timezone_IANA}, {self.tick_interval}"
        return f"{self.start_ms}, {self.end_ms}, {self.timezone_IANA}"

    def __repr__(self) -> str:
        if self.is_discrete:
            return f"Timeframe({self.start.to_datetime()}, {self.end.to_datetime()}, {self.timezone_IANA}, {self.tick_interval})"
        return f"Timeframe({self.start_ms}, {self.end_ms}, {self.timezone_IANA})"

    def __eq__(self, other) -> bool:
        return (self.start_ms == other.start_ms and self.end_ms == other.end_ms
                and self.tick_interval == other.tick_interval)

    def __contains__(self, item: str | int | datetime | KlineTimestamp) -> bool:
        """
        Checks if a given timestamp is within the Timeframe.
        """
        item_ms = _to_ms(item, self.timezone_IANA)
        return self.start_ms <= item_ms <= self.end_ms

    def adjust_to_open(self, inplace: bool = False) -> 'Timeframe':
        """
        Adjusts the start and end to the beginning of their respective tick intervals.
        """
        if not self.is_discrete:
            raise ValueError("Cannot adjust non-discrete timeframe to open.")
        new_start = self.start.open
        new_end = self.end.open
        if inplace:
            self.start = KlineTimestamp(new_start, self.tick_interval, self.timezone_IANA)
            self.end = KlineTimestamp(new_end, self.tick_interval, self.timezone_IANA)
            self.start_ms = new_start
            self.end_ms = new_end
            self.ms = self.end_ms - self.start_ms
            return self
        return Timeframe(new_start, new_end, self.timezone_IANA, self.tick_interval)

    def __add__(self, milliseconds: int) -> 'Timeframe':
        """Moves the timeframe forward by the specified number of milliseconds."""
        return Timeframe(self.start_ms + milliseconds, self.end_ms + milliseconds,
                         self.timezone_IANA, self.tick_interval)

    def __sub__(self, milliseconds: int) -> 'Timeframe':
        """Moves the timeframe backward by the specified number of milliseconds."""
        return Timeframe(self.start_ms - milliseconds, self.end_ms - milliseconds,
                         self.timezone_IANA, self.tick_interval)

    def __len__(self) -> int:
        """
        Returns the number of tick intervals within the timeframe (inclusive), or total milliseconds if not discrete.
        """
        if self.is_discrete:
            return int(self.ms // self.tick_interval_ms) + 1
        return int(self.ms)

    def __gt__(self, other: 'Timeframe') -> bool:
        return len(self) > len(other)

    def __lt__(self, other: 'Timeframe') -> bool:
        return len(self) < len(other)

    def __ge__(self, other: 'Timeframe') -> bool:
        return len(self) >= len(other)

    def __le__(self, other: 'Timeframe') -> bool:
        return len(self) <= len(other)

    def get_ms(self) -> int:
        """Returns the total milliseconds from start to end."""
        return self.ms

    def update_tick_interval(self, tick_interval: str | None, inplace: bool = False):
        """
        Updates the tick interval.

        :param tick_interval: The new tick interval.
        :param inplace: If True, modifies in place. If False, returns new Timeframe.
        """
        if inplace:
            self.tick_interval = tick_interval
            self.tick_interval_ms = tick_seconds[tick_interval] * 1000 if tick_interval else None
            self.is_discrete = bool(tick_interval)
            if self.is_discrete:
                self.start = KlineTimestamp(self.start_ms, tick_interval, self.timezone_IANA)
                self.end = KlineTimestamp(self.end_ms, tick_interval, self.timezone_IANA)
        else:
            return Timeframe(self.start_ms, self.end_ms, self.timezone_IANA, tick_interval)

    def get_hours(self) -> float:
        """
        Returns the number of hours in the Timeframe.
        """
        if self.is_discrete:
            return len(self) * self.tick_interval_ms / (60 * 60 * 1000)
        return self.ms / (60 * 60 * 1000)

    def get_limit(self) -> int:
        """
        Returns the number of candles (len).
        """
        return len(self)

    def _get_start_end_from_hours_or_limit(self,
                                           hours: float | None,
                                           limit: int | None,
                                           start_time=None,
                                           end_time=None) -> tuple[int, int]:
        """
        Calculates start/end from hours or limit.
        """
        assert bool(limit) ^ bool(hours), "Either 'hours' or 'limit' must be specified, but not both or none."

        if hours:
            limit = -(-int(hours * 60 * 60 * 1000) // self.tick_interval_ms)

        now_ms = int(time() * 1000)

        if not end_time and not start_time:
            end_ms = KlineTimestamp(now_ms, self.tick_interval, self.timezone_IANA).prev().open
            start_ms = end_ms - (limit * self.tick_interval_ms)
            return start_ms, end_ms

        elif not end_time and start_time:
            start_ms = _to_ms(start_time, self.timezone_IANA)
            start_ms = KlineTimestamp(start_ms, self.tick_interval, self.timezone_IANA).open
            end_ms = start_ms + (limit * self.tick_interval_ms) - 1
            now_prev = KlineTimestamp(now_ms, self.tick_interval, self.timezone_IANA).prev().open
            end_ms = min(end_ms, now_prev)
            return start_ms, end_ms

        elif end_time and not start_time:
            end_ms = _to_ms(end_time, self.timezone_IANA)
            start_ms = end_ms - (limit * self.tick_interval_ms)
            return start_ms, end_ms

        raise NotImplementedError(f"Case not implemented: start_time={start_time}, end_time={end_time}")

    def __getitem__(self, key):
        """
        Allows indexed access to timestamps within the Timeframe.
        """
        if not self.is_discrete:
            raise TypeError("Indexing is only supported for discrete timeframes.")

        length = len(self)

        if isinstance(key, slice):
            start, stop, step = key.indices(length)
            return [self[i] for i in range(start, stop, step)]

        elif isinstance(key, int):
            if key < 0:
                key += length
            if key < 0 or key >= length:
                raise IndexError("Timeframe index out of range.")
            tick_ms = self.start_ms + (self.tick_interval_ms * key)
            return KlineTimestamp(tick_ms, self.tick_interval, self.timezone_IANA)
        else:
            raise TypeError("Invalid index type. Must be an integer or slice.")

    def __iter__(self):
        """
        Allows iteration over the Timeframe, yielding a KlineTimestamp for each tick interval.
        """
        if not self.is_discrete:
            raise ValueError("Iteration is only supported for discrete timeframes.")
        return _TimeframeIterator(self)

    def get_pandas_index(self, name: str = "") -> pd.DatetimeIndex:
        """
        Returns a pandas DatetimeIndex for the Timeframe.

        :param name: The name of the index.
        """
        start_dt = self.start.to_datetime() if self.is_discrete else datetime.utcfromtimestamp(self.start_ms / 1000)
        end_dt = self.end.to_datetime() if self.is_discrete else datetime.utcfromtimestamp(self.end_ms / 1000)
        return pd.date_range(start=start_dt,
                             end=end_dt,
                             freq=pandas_freq_tick_interval[self.tick_interval],
                             name=name)


class _TimeframeIterator:
    """
    Iterator class for Timeframe.
    """

    def __init__(self, timeframe: Timeframe):
        self._timeframe = timeframe
        self._current_ms = timeframe.start_ms

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_ms > self._timeframe.end_ms:
            raise StopIteration
        kt = KlineTimestamp(self._current_ms, self._timeframe.tick_interval, self._timeframe.timezone_IANA)
        self._current_ms += self._timeframe.tick_interval_ms
        return kt
