from time import time
from typing import Tuple, List
import pandas as pd


from handlers.files import select_file, read_csv_to_dataframe, extract_filename_metadata
from handlers.logs import Logs
from handlers.market import (convert_to_numeric)
from handlers.time_helper import (pandas_freq_tick_interval, open_from_milliseconds, time_interval)
from handlers.wallet import convert_str_date_to_ms

# from handlers.starters import is_running_in_jupyter
# if is_running_in_jupyter():
#     pass
# else:
#     pass

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


def check_continuity(df: pd.DataFrame, time_zone: str) -> pd.DataFrame:
    """
    Verify if the dataframe has continuity in klines by "Open timestamp" column.

    :param df: Dataframe with klines data.
    :param time_zone: A string with the time zone. Ex: 'Europe/Madrid'
    :return: A dataframe with the gaps in klines continuity or if no gaps, returns empty dataframe.
    """
    if df.empty:
        binpan_logger.warning("BinPan Warning: Dataframe is empty.")
        return pd.DataFrame()
    dif = df['Open timestamp'].diff().dropna()  # Drop the NaN value for the first row

    try:
        assert len(dif.value_counts()) == 1, "BinPan Exception: Dataframe has gaps in klines continuity."
        return pd.DataFrame()
    except AssertionError:
        mask = pd.Series(dif != dif.iloc[0]).reindex(df.index).fillna(False)
        gaps = df.loc[mask, ['Open timestamp', 'Close timestamp']]

        # Convertir los timestamps a un formato más legible
        # gaps['Open timestamp'] = to_datetime(gaps['Open timestamp'], unit='ms')
        # gaps['Close timestamp'] = to_datetime(gaps['Close timestamp'], unit='ms')
        gaps['Open timestamp'] = pd.to_datetime(gaps['Open timestamp'], unit='ms', utc=True)
        gaps['Close timestamp'] = pd.to_datetime(gaps['Close timestamp'], unit='ms', utc=True)
        gaps['Open timestamp'] = gaps['Open timestamp'].dt.tz_convert(time_zone)
        gaps['Close timestamp'] = gaps['Close timestamp'].dt.tz_convert(time_zone)

        # También hacer lo mismo para 'dif'
        # dif_readable = to_datetime(dif[mask].index, unit='ms')

        gaps["Gap length"] = gaps["Close timestamp"] - gaps["Open timestamp"]

        binpan_logger.warning(f"BinPan Warning: Dataframe has gaps in klines continuity: \n{gaps}")
        # binpan_logger.warning(f"\nTimestamp discontinuities detected: \n{dif_readable}")
        binpan_logger.info(f"Please, repair the dataframe with the function 'repair_continuity' method or 'repair_kline_discontinuity' "
                           f"function from 'auxiliar.py'.")
        return gaps


def find_common_interval_and_generate_timestamps(data: pd.DataFrame, timestamp_col="Open timestamp") -> Tuple[int, List]:
    """
    Find the most common interval between timestamps and generate a list of timestamps that should be present in the
     dataframe.

    :param data: Dataframe with klines data.
    :param timestamp_col: Name of the column with the timestamps.
    :return: A tuple with the most common interval and a list of timestamps that should be present in the dataframe.
    """
    df = data.copy(deep=True)
    df.sort_values(by=timestamp_col, inplace=True, ascending=True)
    df['Open timestamp diff'] = df[timestamp_col].diff()
    common_interval = int(df['Open timestamp diff'].value_counts().idxmax())

    start_time = int(df[timestamp_col].iloc[0])
    end_time = int(df[timestamp_col].iloc[-1])

    expected_timestamps = list(range(start_time, end_time + common_interval, common_interval))

    return common_interval, expected_timestamps


def create_empty_typed_dataframe(index_data: list,
                                 columns: list,
                                 dtype_dict: dict,
                                 time_zone: str,
                                 index_time_zone: str,
                                 is_index_data_timestamps: bool = True):
    """
    Create an empty DataFrame with the desired types. It uses zeros to avoid NaNs and dtype fixing errors.

    :param index_data: A list with the index data. Can be a list of timestamps or a list of strings.
    :param columns: A list with the column names.
    :param dtype_dict: Types of the columns.
    :param time_zone: A string with the time zone. Ex: 'Europe/Madrid'
    :param index_time_zone: A string with the time zone. Ex: 'Europe/Madrid'
    :param is_index_data_timestamps: If True, the index data will be converted to passed type with this in mind.
    :return:
    """
    df_ = pd.DataFrame(index=index_data, columns=columns)
    df_.fillna(0, inplace=True)

    if is_index_data_timestamps:
        df_.index = pd.to_datetime(df_.index, unit='ms')
        if index_time_zone:
            df_.index = df_.index.tz_localize('UTC').tz_convert(index_time_zone)

    for col, dtype in dtype_dict.items():
        if pd.api.types.is_datetime64_any_dtype(dtype):
            if isinstance(dtype, pd.DatetimeTZDtype):
                # Localizar la zona horaria si el dtype es timezone-aware
                if time_zone:
                    df_[col] = pd.to_datetime(df_[col]).dt.tz_localize('UTC').dt.tz_convert(time_zone).astype(dtype)
                else:
                    df_[col] = pd.to_datetime(df_[col]).dt.tz_localize('UTC').astype(dtype)
            else:
                df_[col] = pd.to_datetime(df_[col]).astype(dtype)
        else:
            df_[col] = df_[col].astype(dtype, errors='ignore')
    # print(type(df_.index))
    # replace zeros with nans
    df_.replace(0, pd.NA, inplace=True)
    return df_.sort_index(ascending=True)


def add_missing_klines(df, expected_timestamps: List[int], timestamp_col="Open timestamp"):
    """
    Add missing rows with nan to the DataFrame based on the list of expected timestamps.

    :param df: DataFrame with financial klines data.
    :param expected_timestamps: List of expected timestamps.
    :param timestamp_col: Name of the column with the timestamps.
    :return: DataFrame with missing rows added.
    """
    df_copy = df.copy(deep=True)
    df_copy.sort_values(by=timestamp_col, inplace=True)
    dtype_dict = df_copy.dtypes.to_dict()

    if df_copy.index.tz:
        time_zone = df_copy.index.tz.zone
        index_time_zone = time_zone
    else:
        time_zone = None
        index_time_zone = None

    missing_timestamps = list(set(expected_timestamps) - set(df_copy[timestamp_col]))
    df_ = create_empty_typed_dataframe(index_data=missing_timestamps,
                                       columns=df_copy.columns,
                                       dtype_dict=dtype_dict,
                                       time_zone=time_zone,
                                       index_time_zone=index_time_zone,
                                       is_index_data_timestamps=True)
    df_[timestamp_col] = sorted(missing_timestamps)

    # index_name = df_copy.index.name
    df_copy = pd.concat([df_copy, df_], axis=0, ignore_index=False)
    # df_copy.index = df_copy['Open time']
    # df_copy.index.name = index_name

    if index_time_zone:
        assert df_.index.tz.zone == df_copy.index.tz.zone, "BinPan Exception: Time zones are not the same."

    df_copy.sort_values(by=timestamp_col, inplace=True)
    assert len(expected_timestamps) == len(df_copy), "BinPan Exception: There are still missing values in the DataFrame."
    return df_copy


def fill_missing_values(data: pd.DataFrame, interval_ms: int, time_zone: str) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.

    :param data: DataFrame with financial klines data.
    :param interval_ms: Interval between timestamps in milliseconds.
    :param time_zone: A string with the time zone. Ex: 'Europe/Madrid'
    :return: DataFrame with missing values filled.
    """
    df = data.copy(deep=True)
    price_cols = ['Open', 'High', 'Low', 'Close']
    volume_cols = ['Volume', 'Quote volume', 'Trades', 'Taker buy base volume', 'Taker buy quote volume']

    df_filled = df.copy(deep=True)

    df_filled[price_cols] = df_filled[price_cols].ffill()
    df_filled[volume_cols] = df_filled[volume_cols].fillna(0)

    if "First TradeId" in df_filled.columns:
        missing_first_trade_id = df_filled['First TradeId'].isna()
        # ffill last_trade_id column
        df_filled['Last TradeId'] = df_filled['Last TradeId'].ffill()
        df_filled.loc[missing_first_trade_id, 'First TradeId'] = df_filled.loc[missing_first_trade_id]['Last TradeId']

    # obtain date columns from timestamp columns
    df_filled['Close timestamp'] = df['Open timestamp'] + interval_ms - 1
    df_filled['Open time'] = pd.to_datetime(df_filled['Open timestamp'], unit='ms', utc=True)
    df_filled['Close time'] = pd.to_datetime(df_filled['Close timestamp'], unit='ms', utc=True)

    # convert timestamp columns to datetime
    if time_zone:
        df_filled['Open time'] = df_filled['Open time'].dt.tz_convert(time_zone)
        df_filled['Close time'] = df_filled['Close time'].dt.tz_convert(time_zone)

    return df_filled


def repair_kline_discontinuity(df: pd.DataFrame, time_zone: str) -> pd.DataFrame:
    """
    Repair kline discontinuity in the DataFrame. Price columns are filled with the previous value and volume columns are
     filled with 0. Timestamps are also filled with the missing values inferred from the most common interval. Dates columns
     are also filled with the missing values inferred from the most common interval.

    :param df: A DataFrame with klines data.
    :param time_zone: A string with the time zone. Ex: 'Europe/Madrid'
    :return: Repaired DataFrame.
    """
    binpan_logger.info("Repairing kline discontinuity with selective ffill...")
    interval, expected_ts = find_common_interval_and_generate_timestamps(df)
    binpan_logger.info(f"Common interval found: {interval} ms. Inserting {len(expected_ts) - len(df)} missing rows...")
    df_filled = add_missing_klines(df=df, expected_timestamps=expected_ts)
    binpan_logger.info("Filling missing values...")
    ret = fill_missing_values(data=df_filled, interval_ms=interval, time_zone=time_zone)
    ret.index.name = df.index.name
    return ret
