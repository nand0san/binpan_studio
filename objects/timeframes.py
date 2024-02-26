
from datetime import datetime
import pytz


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


class KlineTimeframe:
    """
    Calculates the discrete timeframes for retrieving klines data from the Binance API. The timeframes are defined by
    the start and end times, and the kline interval. The start and end times are provided as strings in the format
    "%Y-%m-%d %H:%M:%S" and are adjusted to the specified time zone. The kline interval is provided as a string, e.g.,
    "1m", "1h", "1d", etc. The timeframes are calculated in milliseconds since epoch (1 Jan 1970).

    """
    tick_milliseconds = tick_milliseconds

    def __init__(self,
                 interval: str,
                 start_time: str = None,
                 end_time: str = None,
                 time_zone_str: str = 'UTC',
                 hours: int = None,
                 limit: int = 1000):
        """
        Initializes the KlineTimeframe object. It completes the startime and endtime data based on the provided
        parameters.

        - If hours is provided, the start_time is adjusted to the current time minus the specified number of hours. Hours
          option is prevalent over start_time and end_time.
        - If start_time and end_time are not provided, the default is to retrieve the last 1000 klines.
        - If not end_time is provided but start_time is, the end_time is calculated by the limit of klines to retrieve. Standard
          limit of 1000 klines is ignored in this case.
        - If not start_time is provided but end_time is, the start_time is calculated by the limit of klines to retrieve. Standard
          limit of 1000 klines is ignored in this case.
        -  If start_time and end_time are provided, the limit is ignored and the timeframe is set by the start and end times.

        In kline options:
        - In case of a startime is inside any kline timeframe, the kline will be included in the retrieval.
        - In case of a endtime is inside any kline timeframe, the kline will be included in the retrieval.

        :param interval: Kline interval.
        :param start_time: Start time as a string.
        :param end_time: End time as a string.
        :param time_zone_str: Time zone for interpreting start and end times.
        :param hours: Limit the data by hours.
        :param limit: Maximum number of klines to retrieve.
        """
        self.interval = interval
        self.interval_millis = tick_milliseconds[self.interval]
        self.time_zone_str = time_zone_str
        self.limit = limit
        self.hours = hours

        self.timezone = pytz.timezone(time_zone_str)
        self._calculate_timeframes(start_time, end_time)

    def _str_to_datetime(self, date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=self.timezone)

    def _datetime_to_millis(self, date_time: datetime) -> int:
        return convert_str_date_to_ms(date=date_time, time_zone=self.time_zone_str)

    def _millis_to_datetime(self, millis: int) -> datetime:
        return datetime.fromtimestamp(millis / 1000, tz=self.timezone)

    def _adjust_time_to_interval(self, timestamp_ms: int) -> int:
        # Calcula el número de unidades de intervalo en el timestamp dado
        print(f"- timestamp_ms: {timestamp_ms}, interval_millis: {self.interval_millis}")
        units = timestamp_ms // self.interval_millis
        adjusted_timestamp = units * self.interval_millis
        print(f"- units: {units}, adjusted_timestamp: {adjusted_timestamp}")
        return adjusted_timestamp

    def _calculate_timeframes(self, start_time, end_time):
        now = datetime.now(pytz.utc)
        now_millis = self._datetime_to_millis(now)

        if self.hours is not None:
            self.end_time = now_millis
            self.start_time = self.end_time - self.hours * 60 * 60 * 1000
        else:
            if start_time is not None and end_time is not None:
                self.start_time = self._datetime_to_millis(self._str_to_datetime(start_time))
                self.end_time = self._datetime_to_millis(self._str_to_datetime(end_time))
                print(self._str_to_datetime(start_time))
                print(self._str_to_datetime(end_time))
            elif start_time is not None and end_time is None:
                self.start_time = self._datetime_to_millis(self._str_to_datetime(start_time))
                self.end_time = self.start_time + self.interval_millis * self.limit
            elif start_time is None and end_time is not None:
                self.end_time = self._datetime_to_millis(self._str_to_datetime(end_time))
                self.start_time = self.end_time - self.interval_millis * self.limit
            else:
                self.start_time = now_millis - self.interval_millis * self.limit
                self.end_time = now_millis

        print(f"start_time: {self._millis_to_datetime(self.start_time)}, start_time: {self.start_time}")
        print(f"end_time: {self._millis_to_datetime(self.end_time)}, end_time: {self.end_time}")
        print(f"segundos de diferencia entre start y end: {(self.end_time - self.start_time) / 1000}")
        # Adjust start and end times to the interval
        self.start_time = self._adjust_time_to_interval(self.start_time)
        self.end_time = self._adjust_time_to_interval(self.end_time)

        print(f"start_time: {self._millis_to_datetime(self.start_time)}, start_time: {self.start_time}")
        print(f"end_time: {self._millis_to_datetime(self.end_time)}, end_time: {self.end_time}")
        print(f"segundos de diferencia entre start y end: {(self.end_time - self.start_time) / 1000}")

    def get_timeframes(self):
        """
        Returns the adjusted start and end times in epoch milliseconds.
        """
        return self.start_time, self.end_time


####################
# Helper Functions #
####################


def convert_str_date_to_ms(date, time_zone: str) -> int:
    """
    Converts a date (either a string in the format "%Y-%m-%d %H:%M:%S" or a datetime object)
    to milliseconds considering the time zone.

    :param date: Date to be converted.
    :param time_zone: Time zone string, e.g., "UTC", "Europe/Madrid".
    :return: Time in milliseconds since epoch (1 Jan 1970) adjusted for the specified time zone.
    """
    tz = pytz.timezone(time_zone)

    # Check if the date is already an integer (milliseconds since epoch)
    if isinstance(date, int):
        return date

    # If date is a string, convert it to a datetime object
    if isinstance(date, str):
        dt_naive = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        dt_aware = tz.localize(dt_naive)
    elif isinstance(date, datetime):
        # If date is a datetime object, localize it to the specified time zone if it's not already
        if date.tzinfo is None:
            dt_aware = tz.localize(date)
        else:
            dt_aware = date.astimezone(tz)
    else:
        raise ValueError("The date must be a string, datetime, or integer.")

    # Convert the timezone-aware datetime object to UTC and then to milliseconds since epoch
    dt_utc = dt_aware.astimezone(pytz.utc)
    millis_since_epoch = int(dt_utc.timestamp() * 1000)

    return millis_since_epoch


if __name__ == "__main__":
    # Example usage
    kline_timeframe = KlineTimeframe(interval="1h",
                                     start_time="2021-01-01 00:00:00",
                                     end_time="2021-01-01 00:02:00",
                                     time_zone_str="Europe/Madrid")

    start_time, end_time = kline_timeframe.get_timeframes()
    print(f"Start time: {start_time}, End time: {end_time}")
    print(f"segundos de diferencia entre start y end: {(end_time - start_time) / 1000}")
    print(f"horas de diferencia entre start y end: {(end_time - start_time) / 1000 / 3600}")
    print(f"dias de diferencia entre start y end: {(end_time - start_time) / 1000 / 3600 / 24}")
