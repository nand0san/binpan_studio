from time import time

import pandas as pd

from handlers.files import select_file, read_csv_to_dataframe, extract_filename_metadata
from handlers.logs import Logs
from handlers.market import (convert_to_numeric)
from handlers.starters import is_running_in_jupyter
from handlers.time_helper import (pandas_freq_tick_interval, open_from_milliseconds, time_interval)
from handlers.wallet import convert_str_date_to_ms

if is_running_in_jupyter():
    pass
else:
    pass

binpan_logger = Logs(filename='./logs/binpan.log', name='binpan', info_level='INFO')


def csv_klines_setup(from_csv: str or bool,
                     symbol: str,
                     tick_interval: str,
                     cwd: str,
                     time_zone: str, ) -> tuple:
    """
    Loads a csv file with klines data and returns a tuple with the data and metadata.

    :param from_csv: Can be a string with the path to the csv file or a boolean. If boolean, it will search for the file in the cwd.
    :param symbol: Expected symbol as filter.
    :param tick_interval: Tick interval as filter.
    :param cwd: Working directory.
    :param time_zone: A string with the time zone. Ex: 'Europe/Madrid'
    :return: Basic metadata and the dataframe.
    """

    if type(from_csv) == str:
        filename = from_csv
    else:
        filename = select_file(path=cwd,
                               extension='csv',
                               symbol=symbol,
                               tick_interval=tick_interval,
                               name_filter='klines')
    binpan_logger.info(f"Loading {filename}")

    # load and to numeric types
    df = read_csv_to_dataframe(filename=filename, index_col="Open timestamp", index_time_zone=time_zone)
    df = convert_to_numeric(data=df)

    # basic metadata
    symbol, tick_interval, time_zone, data_type, start_time, end_time = extract_filename_metadata(filename=filename,
                                                                                                  expected_data_type="klines",
                                                                                                  expected_symbol=symbol,
                                                                                                  expected_interval=tick_interval,
                                                                                                  expected_timezone=time_zone)
    symbol = symbol.upper()
    df.index.name = f"{symbol} {tick_interval} {time_zone}"

    if not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(df['Open timestamp'], unit='ms')).tz_localize('UTC').tz_convert(time_zone)
        df.index = idx
    my_pandas_freq = pandas_freq_tick_interval.get(tick_interval)
    prev_len = len(df)
    df = df.asfreq(my_pandas_freq)  # this adds freq, will add nans if missing api data in the middle
    if prev_len != len(df):
        filled_data_by_freq = df[df.isna().all(axis=1)]
        binpan_logger.warning(f"Missing data in {filename} for {symbol} {tick_interval} {time_zone}: \n Filled "
                              f"data automatically: \n{filled_data_by_freq}")
    limit = len(df)

    last_timestamp_ind_df = df.iloc[-1]['Close timestamp']
    if last_timestamp_ind_df >= int(time() * 1000):
        closed = False
    else:
        closed = True

    return df, symbol, tick_interval, time_zone, data_type, start_time, end_time, closed, limit


def setup_startime_endtime(start_time: str,
                           end_time: str,
                           time_zone: str,
                           hours: int,
                           closed: bool,
                           tick_interval: str,
                           limit: int):
    start_time = convert_str_date_to_ms(date=start_time, time_zone=time_zone)
    end_time = convert_str_date_to_ms(date=end_time, time_zone=time_zone)

    # work with open timestamps
    if start_time:
        start_time = open_from_milliseconds(ms=start_time, tick_interval=tick_interval)
    if end_time:
        end_time = open_from_milliseconds(ms=end_time, tick_interval=tick_interval)

    # limit by hours
    if hours:
        if not end_time:
            end_time = int(time() * 1000)
        if not start_time:
            start_time = end_time - (hours * 60 * 60 * 1000)

    # fill missing timestamps
    start_time, end_time = time_interval(tick_interval=tick_interval,
                                         limit=limit,
                                         start_time=start_time,
                                         end_time=end_time)
    # discard not closed
    now = int(1000 * time())
    current_open = open_from_milliseconds(ms=now, tick_interval=tick_interval)
    if closed and end_time >= current_open:
        end_time = current_open - 2000
    return start_time, end_time


def check_continuity(df: pd.DataFrame):
    """
    Verify if the dataframe has continuity in klines by "Open timestamp" column.
    """

    dif = df['Open timestamp'].diff().dropna()  # Drop the NaN value for the first row

    try:
        assert len(dif.value_counts()) == 1, "BinPan Exception: Dataframe has gaps in klines continuity."
    except AssertionError:
        # he añadido lo de pd series y podría fallar v0.5.0
        mask = pd.Series(dif != dif.iloc[0]).reindex(df.index).fillna(False)
        gaps = df.loc[mask, ['Open timestamp', 'Close timestamp']]
        binpan_logger.warning(f"BinPan Warning: Dataframe has gaps in klines continuity: \n{gaps}")
        binpan_logger.warning(f"\nTimestamp differences detected: \n{dif[mask]}")
