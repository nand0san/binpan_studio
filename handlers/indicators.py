"""

BinPan own indicators and utils.

"""
import pandas as pd
import pytz
import numpy as np
from typing import Tuple, List
import os
import multiprocessing

# from .starters import is_running_in_jupyter
from .time_helper import convert_milliseconds_to_time_zone_datetime
from .time_helper import pandas_freq_tick_interval
from .tags import is_alternating

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm

# this is to avoid the error: "RuntimeError: can't set attribute" when using multiprocessing
cpus = multiprocessing.cpu_count() // 2
os.environ["LOKY_MAX_CPU_COUNT"] = str(cpus)


##############
# INDICATORS #
##############


def alternating_fractal_indicator(df: pd.DataFrame, max_period: int = None, suffix: str = "") -> pd.DataFrame or None:
    """
    Obtains the minim value for fractal_w periods as fractal is pure alternating max to mins. In other words, max and mins alternates
     in regular rhythm without any tow max or two mins consecutively.

    This custom indicator shows the minimum period in finding a pure alternating fractal. It is some kind of volatility in price
     indicator, the most period needed, the most volatile price.

    :param pd.DataFrame df: BinPan Symbol dataframe.
    :param int max_period: Default is len of df. This method will check from 2 to the max period value to find a puer alternating max to
     mins.
    :param str suffix: A decorative suffix for the name of the column created.
    :return pd.DataFrame: A dataframe with two columns, one with 1 or -1 for local max or local min to tag, and other with price values for
     that points. If not found, it will return None.

    .. image:: images/indicators/fractal_w.png
        :width: 1000
    """
    fractal = None
    if not max_period:
        max_period = len(df)
    print("Searching for pure alternating fractal...")
    for i in tqdm(range(2, max_period)):
        fractal = fractal_w_indicator(df=df, period=i, suffix=suffix, fill_with_zero=True)
        max_min = fractal[f"Fractal_W_{i}"].fillna(0)
        if is_alternating(lst=max_min.tolist()):
            print(f"Pure alternating fractal found at period={i}")
            break
        else:
            fractal = None
    return fractal


def fractal_trend_indicator(df: pd.DataFrame,
                            period: int = None,
                            fractal: pd.DataFrame = None,
                            suffix: str = "") -> tuple or None:
    """
    Obtains the trend of the fractal_w indicator. It will return maximums diff mean and minimums diff mean also in a tuple.

    :param pd.DataFrame df: BinPan Symbol dataframe.
    :param int period: Period to obtain fractal_w. Default is len df.
    :param pd.DataFrame fractal: Optional. If not provided, it will be calculated.
    :param str suffix: A decorative suffix for the name of the column created.
    :return tuple: Max min diffs mean and Min diffs mean.

    .. image:: images/indicators/fractal_w.png
        :width: 1000
    """
    if not period:
        period = len(df)
    if type(fractal) != pd.DataFrame:
        fractal = alternating_fractal_indicator(df=df, max_period=period, suffix=suffix)
    if type(fractal) != pd.DataFrame:
        return
    max_min_col, values_col = fractal.columns[0], fractal.columns[1]
    max_prices = fractal.loc[fractal[max_min_col] == 1][values_col].reindex(df.index).ffill()
    min_prices = fractal.loc[fractal[max_min_col] == -1][values_col].reindex(df.index).ffill()
    max_trend = max_prices.diff().replace(0, np.nan)
    min_trend = min_prices.diff().replace(0, np.nan)
    print(f"Maximum_max_diff={max_trend.max()}")
    print(f"Minimum_max_diff={max_trend.min()}")
    print(f"Maximum_min_diff={min_trend.max()}")
    print(f"Minimum_min_diff={min_trend.min()}")
    return max_trend.mean(), min_trend.mean()


def calculate_fractal_trend_on_flags(df: pd.DataFrame, flags: pd.Series, period: int = None, suffix: str = "") -> List[tuple]:
    """
    Applies the fractal_trend_indicator function to the dataframe df for each index flagged with 1 in the flags series.

    This function is designed to work with a DatetimeIndex.

    :param pd.DataFrame df: BinPan Symbol dataframe.
    :param pd.Series flags: Series containing flags (1 for indices to include).
    :param int period: Number of periods (rows) to look back for calculating fractal trend. Defaults to length of df.
    :param str suffix: A decorative suffix for the name of the column created.
    :return: A list of tuples with the results of fractal_trend_indicator for each flagged index.
    """
    results = []
    # Ensure flags index aligns with df index
    # flags = flags.reindex(df.index, fill_value=0)
    flagged_indices = flags[flags == 1].index

    for flag_date in flagged_indices:
        print(f"\nCalculating fractal trend for {flag_date}")
        # Find the starting index for the period
        start_period_index = df.index.get_loc(flag_date) - period
        if start_period_index < 0:
            continue

        # Obtain the molecule dataframe based on the current index and period
        molecule_df = df.iloc[start_period_index:df.index.get_loc(flag_date)]
        # Calculate the fractal trend indicator for this molecule
        trend_result = fractal_trend_indicator(df=molecule_df, period=period, suffix=suffix)
        results.append((flag_date, trend_result))

    return results


def ichimoku(data: pd.DataFrame, tenkan: int = 9, kijun: int = 26, chikou_span: int = 26, senkou_cloud_base: int = 52,
             suffix: str = '', ) -> pd.DataFrame:
    """
    The Ichimoku Cloud is a collection of technical indicators that show support and resistance levels, as well as momentum and trend
    direction. It does this by taking multiple averages and plotting them on a chart. It also uses these figures to compute a “cloud”
    that attempts to forecast where the price may find support or resistance in the future.

    https://school.stockcharts.com/doku.php?id=technical_indicators:ichimoku_cloud

    https://www.youtube.com/watch?v=mCri-FFvZjo&list=PLv-cA-4O3y97HAd9OCvVKSfvQ8kkAGKlf&index=7

    :param pd.DataFrame data: A BinPan Symbol dataframe.
    :param int tenkan: The short period. It's the half sum of max and min price in the window. Default: 9
    :param int kijun: The long period. It's the half sum of max and min price in the window. Default: 26
    :param int chikou_span: Close of the next 26 bars. Util when spotting what happened with other ichimoku lines and what happened
      before Default: 26.
    :param senkou_cloud_base: Period to obtain kumo cloud base line. Default is 52.
    :param str suffix: A decorative suffix for the name of the column created.
    :return pd.DataFrame: A pandas dataframe with columns as indicator lines.

       .. code-block:: python

            import binpan

            sym = binpan.Symbol(symbol='LUNCBUSD', tick_interval='1m', limit=500)
            sym.ichimoku()
            sym.plot()

       .. image:: images/indicators/ichimoku.png
        :width: 1000
        :alt: Candles with some indicators

    """
    df = data.copy(deep=True)
    my_timezone = df.index.name.split()[-1]
    tick_interval = df.index.name.split()[1]

    if df.index.tz is None:
        df.index = df.index.tz_localize(pytz.UTC, ambiguous='infer')
    else:
        df.index = df.index.tz_convert("UTC")

    if not df.index.freq:
        df = df.asfreq(pandas_freq_tick_interval[tick_interval])

    high = df['High']
    low = df['Low']
    close = df['Close']

    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2

    chikou_span_serie = close.shift(periods=-chikou_span, freq='infer')

    arr_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_a = arr_a.shift(periods=chikou_span, freq='infer')

    arr_b = (high.rolling(window=senkou_cloud_base).max() + low.rolling(window=senkou_cloud_base).min()) / 2
    senkou_span_b = arr_b.shift(periods=chikou_span, freq='infer')

    ret = pd.DataFrame([tenkan_sen, kijun_sen, chikou_span_serie, senkou_span_a, senkou_span_b]).T
    ret.index = ret.index.tz_convert(my_timezone)

    if suffix:
        suffix = '_' + suffix

    ret.columns = [f"Ichimoku_tenkan_{tenkan}" + suffix, f"Ichimoku_kijun_{kijun}" + suffix, f"Ichimoku_chikou_span{chikou_span}" + suffix,
                   f"Ichimoku_cloud_{tenkan}_{kijun}" + suffix, f"Ichimoku_cloud_{senkou_cloud_base}" + suffix]
    return ret


def ker(close: pd.Series, window: int, ) -> pd.Series:
    """
    Kaufman's Efficiency Ratio based in: https://stackoverflow.com/questions/36980238/calculating-kaufmans-efficiency-ratio-in-python
    -with-pandas

    :param pd.Series close: Close prices serie.
    :param int window: Window to check indicator.
    :return pd.Series: Results.
    """
    direction = close.diff(window).abs()
    # noinspection PyUnresolvedReferences
    volatility = pd.rolling_sum(close.diff().abs(), window)
    return direction / volatility


def fractal_w_indicator(df: pd.DataFrame, period=2, merged: bool = True, suffix: str = '', fill_with_zero: bool = None, ) -> pd.DataFrame:
    """
    The fractal indicator is based on a simple price pattern that is frequently seen in financial markets. Outside of trading, a fractal
    is a recurring geometric pattern that is repeated on all time frames. From this concept, the fractal indicator was devised.
    The indicator isolates potential turning points on a price chart. It then draws arrows to indicate the existence of a pattern.

        https://www.investopedia.com/terms/f/fractal.asp

        From: https://codereview.stackexchange.com/questions/259703/william-fractal-technical-indicator-implementation

    :param pd.DataFrame df: BinPan dataframe with High prices.
    :param int period: Default is 2. Count of neighbour candles to match max or min tags.
    :param bool merged: If True, values are merged into one pd.Serie. minimums overwrite maximums in case of coincidence.
    :param str suffix: A decorative suffix for the name of the column created.
    :param bool fill_with_zero: If true fills nans with zeros. Its better to plot with binpan.
    :return pd.DataFrame: A dataframe with two columns, one with 1 or -1 for local max or local min to tag, and other with price values for
     that points.

    .. image:: images/indicators/fractal_w.png
        :width: 1000
    """
    df = df.copy(deep=True)
    window = 2 * period + 1  # default 5

    mins = df['Low'].rolling(window, center=True).apply(lambda x: x[period] == min(x), raw=True)
    maxs = df['High'].rolling(window, center=True).apply(lambda x: x[period] == max(x), raw=True)

    mins = mins.replace({0: np.nan})
    maxs = maxs.replace({0: np.nan})
    # maxs.loc[:] = maxs * -1
    mins.loc[:] = mins * -1

    # return mins, maxs
    if suffix:
        suffix = '_' + suffix

    mins.name = f"Fractal_W_Min_{period}" + suffix
    maxs.name = f"Fractal_W_Max_{period}" + suffix

    min_idx = mins.dropna().index
    max_idx = maxs.dropna().index

    values = pd.Series(np.nan, index=df.index)
    values.loc[min_idx] = df['Low']
    values.loc[max_idx] = df['High']
    values.name = f"Fractal_W_{period}_values" + suffix

    if not merged:
        if fill_with_zero:
            mins.fillna(0, inplace=True)
            maxs.fillna(0, inplace=True)
        return pd.DataFrame([mins, maxs, values]).T

    else:
        merged = mins
        merged.fillna(maxs, inplace=True)
        merged.name = f"Fractal_W_{period}" + suffix
        if fill_with_zero:
            merged.fillna(0, inplace=True)
        return pd.DataFrame([merged, values]).T


def support_resistance_volume(df, num_bins=100, price_col='Close', volume_col='Volume', threshold=90):
    """
    Calculates support and resistance levels based on volume and prices in the given dataframe.

    :param pd.DataFrame df: The dataframe containing price and volume data.
    :param int num_bins: Optional. The number of bins to use for accumulating volume. Default is 100.
    :param str price_col: Optional. The name of the column containing price data. Default is 'Close'.
    :param str volume_col: Optional. The name of the column containing volume data. Default is 'Volume'.
    :param int threshold: Percentil to show most traded levels. Default is 90. The threshold variable filters the volume levels
     considered when calculating support and resistance levels. A higher threshold results in fewer levels, based on higher transaction
     volumes, while a lower threshold yields more levels, based on a broader range of transaction volumes.
    :return list: A sorted list of support and resistance levels.
    """

    # Calculate the price range
    max_price = df[price_col].max()
    min_price = df[price_col].min()
    price_range = max_price - min_price

    # Create bins for accumulating volume
    bins = np.linspace(min_price, max_price, num=num_bins + 1)
    bin_volumes = np.zeros(num_bins)

    # Accumulate volume in each bin
    for _, row in df.iterrows():
        price = row[price_col]
        volume = row[volume_col]
        bin_index = int((price - min_price) // (price_range / num_bins))
        bin_volumes[bin_index] += volume

    # Find the support and resistance levels
    support_resistance_levels = []
    for idx, volume in enumerate(bin_volumes):
        if volume >= np.percentile(bin_volumes, threshold):  # Customize the threshold according to preference
            level = (bins[idx] + bins[idx + 1]) / 2
            support_resistance_levels.append(level)

    return sorted(support_resistance_levels)


##############################
# HIGH OF THE DAY INDICATORS #
##############################


def count_smaller_values_backward(s: pd.Series or list) -> pd.Series:
    """
    Calcula el número de valores que cada entrada en la serie supera hacia atrás
    hasta encontrar un valor superior o llegar al inicio de la serie.

    :param s: Serie de pandas o lista de números.
    :return: Serie de pandas con el número de valores que cada entrada supera hacia atrás.
    """
    if isinstance(s, list):
        s = pd.Series(s)

    # Inicializar una serie para almacenar los resultados
    result = pd.Series(index=s.index, dtype=int)

    # Calcular el número de valores que cada entrada supera hacia atrás
    for i in range(len(s)):
        count = 0
        for j in range(i - 1, -1, -1):
            if s.iloc[j] > s.iloc[i]:
                break
            count += 1
        result.iloc[i] = count

    return result


def count_larger_values_backward(s: pd.Series or list) -> pd.Series:
    """
    Calcula el número de valores que son superiores a cada entrada en la serie hacia atrás
    hasta encontrar un valor igual o menor o llegar al inicio de la serie.

    :param s: Serie de pandas o lista de números.
    :return: Serie de pandas con el número de valores que son superiores a cada entrada hacia atrás.
    """
    if isinstance(s, list):
        s = pd.Series(s)

    # Inicializar una serie para almacenar los resultados
    result = pd.Series(index=s.index, dtype=int)

    # Calcular el número de valores superiores a cada entrada hacia atrás
    for i in range(len(s)):
        count = 0
        for j in range(i - 1, -1, -1):
            if s.iloc[j] <= s.iloc[i]:
                break
            count += 1
        result.iloc[i] = count

    return result


def rolling_max_with_steps_back(ser: pd.Series, window: int, pct_diff: bool = True) -> (pd.Series, pd.Series):
    """
    Calculate the rolling maximum and the steps back to the maximum for each point in the series.

    Parameters:
    series (pd.Series): The time series of prices.
    window (int): The rolling window size.
    pct_result (bool): If True, the maximum values are returned as percentages instead of absolute values (default True).

    Returns:
    pd.Series: A series of the rolling maximum values.
    pd.Series: A series indicating the number of steps back to the maximum value within the window.
    """
    if pct_diff:
        # rolling_max = ser.rolling(window=window).max() / ser - 1
        rolling_max = ser / ser.rolling(window=window).max() - 1
    else:
        rolling_max = ser.rolling(window=window).max()
    max_indices = ser.rolling(window=window).apply(lambda x: np.where(x == x.max())[0][-1], raw=False)
    steps_back = window - 1 - max_indices
    return rolling_max, steps_back


def rolling_min_with_steps_back(ser: pd.Series, window: int, pct_diff: bool = True) -> (pd.Series, pd.Series):
    """
    Calculate the rolling minimum and the steps back to the minimum for each point in the series.

    Parameters:
    series (pd.Series): The time series of prices.
    window (int): The rolling window size.
    pct_result (bool): If True, the minimum values are returned as percentages instead of absolute values (default True).

    Returns:
    pd.Series: A series of the rolling minimum values.
    pd.Series: A series indicating the number of steps back to the minimum value within the window.
    """
    if pct_diff:
        # rolling_min = series.rolling(window=window).min() / series - 1
        rolling_min = ser / ser.rolling(window=window).min() - 1
    else:
        rolling_min = ser.rolling(window=window).min()
    min_indices = ser.rolling(window=window).apply(lambda x: np.where(x == x.min())[0][-1], raw=False)
    steps_back = window - 1 - min_indices
    return rolling_min, steps_back


####################
# INDICATORS UTILS #
####################

def split_serie_by_position(serie: pd.Series, splitter_serie: pd.Series, fill_with_zeros: bool = True) -> pd.DataFrame:
    """
    Splits a serie by values of other serie in four series by relative positions for plotting colored clouds with plotly.

    This means you will get 4 series with different situations:

       - serie is over the splitter serie.
       - serie is below the splitter serie.
       - splitter serie is over the serie.
       - splitter serie is below the serie.

    :param pd.Series serie: A serie to classify in reference to other serie.
    :param pd.Series splitter_serie: A serie to split in two couple of series classified by position reference.
    :param bool fill_with_zeros: Fill nans with zeros for splitted lines like MACD to avoid artifacts in plots.
    :return tuple: A tuple with four series classified by upper position or lower position.
    """
    serie_up = serie.loc[serie.ge(splitter_serie)]
    # splitter_up = splitter_serie.loc[serie_up.index]
    splitter_up = splitter_serie.loc[serie.ge(splitter_serie)]

    serie_down = serie.loc[serie.lt(splitter_serie)]
    # splitter_down = splitter_serie.loc[serie_down.index]
    splitter_down = splitter_serie.loc[serie.lt(splitter_serie)]

    serie_up.name += '_up'
    serie_down.name += '_down'
    splitter_up.name += '_split_up'
    splitter_down.name += '_split_down'

    # return serie_up, serie_splitter_up, serie_down, serie_splitter_down
    # return serie_up, serie_down
    empty_df = pd.DataFrame(index=serie.index)

    empty_df.loc[serie_up.index, serie_up.name] = serie_up
    empty_df.loc[splitter_up.index, splitter_up.name] = splitter_up

    empty_df.loc[serie_down.index, serie_down.name] = serie_down
    empty_df.loc[splitter_down.index, splitter_down.name] = splitter_down

    # ret = pd.DataFrame([serie_up, splitter_up, serie_down, splitter_down]).T

    if fill_with_zeros:
        return empty_df.fillna(0)
    else:
        return empty_df


def df_splitter(data: pd.DataFrame, up_column: str, down_column: str) -> list:
    """
    Splits a dataframe y sub dataframes to plot by colored areas.

    Based on: https://stackoverflow.com/questions/64741015/plotly-how-to-color-the-fill-between-two-lines-based-on-a-condition

    :param pd.DataFrame data: Indicator Dataframe
    :param str up_column: Name of the column to plot green when up side.
    :param str down_column: Name of the column to plot green when down side.
    :return list: A list with splitted dataframes to plot for.
    """

    df = data.copy(deep=True)

    # split data into chunks where series cross each other
    df['label'] = np.where(df[up_column].gt(df[down_column]), 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()
    df = df.groupby('group')
    dfs = []

    for name, data in df:
        dfs.append(data)
    return dfs


def zoom_cloud_indicators(plot_splitted_serie_couples: dict, main_index: list, start_idx: int, end_idx: int) -> dict:
    """
    It zooms the cloud indicators in an index interval for a plot zoom.

    :param dict plot_splitted_serie_couples: Splitted indicators for cloud colored area plotting.
    :param list main_index: The BinPan general index for cutting.
    :param int start_idx: A index to cut.
    :param int end_idx: A index to cut.
    :return dict: All indicators cut.
    """
    try:
        if end_idx:
            assert start_idx < end_idx <= len(main_index)
    except AssertionError:
        raise Exception(f"BinPan Plot Error: Zoom index not valid. Not start={start_idx} < end={end_idx} < len={len(main_index)}")

    ret = {}
    my_start = main_index[start_idx]
    if end_idx:
        my_end = main_index[end_idx]
    else:
        my_end = None
    for k, v in plot_splitted_serie_couples.items():
        splits = v[2]
        cut_splits = []
        for df in splits:
            result = df.loc[my_start:my_end]
            if not result.empty:
                cut_splits.append(result)

        ret[k] = [v[0], v[1], cut_splits, v[3], v[4]]
    return ret


def shift_indicator(serie: pd.Series, window: int = 1):
    """
    It shifts a candle ahead by the window argument value (or backwards if negative).

    Just works with time indexes.

    :param pd.Series serie: A pandas Series.
    :param int window: Times values are shifted ahead. Default is 1.
    :return pd.Series: A series with index adjusted to the new shifted positions of values.
    """
    return serie.shift(window, freq='infer')


def ffill_indicator(serie: pd.Series, window: int = 1):
    """
    It forward fills a value through nans while a window of candles ahead.

    :param pd.Series serie: A pandas Series.
    :param int window: Times values are shifted ahead. Default is 1.
    :return pd.Series: A series with index adjusted to the new shifted positions of values.
    """
    return serie.ffill(limit=window)


###############
# From trades #
###############

def reversal_candles(trades: pd.DataFrame, decimal_positions: int, time_zone: str, min_height: int = 7,
                     min_reversal: int = 4) -> pd.DataFrame:
    """
    Generate reversal candles for reversal charts:
       https://atas.net/atas-possibilities/charts/how-to-set-reversal-charts-for-finding-the-market-reversal/

    :param pd.Series trades: A dataframe with trades sizes, side and prices.
    :param int decimal_positions: Because this function uses integer numbers for prices, is needed to convert prices. Just steps are
     relevant.
    :param str time_zone: A time zone like "Europe/Madrid".
    :param int min_height: Minimum candles height in pips.
    :param int min_reversal: Maximum reversal to close the candles
    :return pd.DataFrame: A serie with the resulting candles number sequence.

     Example:
        .. code-block:: python

           import binpan

            ltc = binpan.Symbol(symbol='ltcbtc',
                                tick_interval='5m',
                                time_zone = 'Europe/Madrid',
                                time_index = True,
                                closed = True,
                                hours=5)
           ltc.get_trades()
           ltc.get_reversal_candles()
           ltc.plot_reversal()

        .. image:: images/indicators/reversal.png
           :width: 1000

    """

    prices = (trades['Price'].to_numpy() * 10 ** decimal_positions).astype(int)
    height = 0

    candle = 0
    prev_candle = 0
    candle_ids = [0]  # first value in a candle

    current_open = prices[0]
    open_ = [current_open]
    high = [current_open]
    low = [current_open]
    close = [current_open]
    prices_pool = [current_open]
    # reversal_control = [0]

    for idx in range(1, prices.size):
        current_price = prices[idx]
        previous_price = prices[idx - 1]
        delta = current_price - previous_price

        if candle != prev_candle:
            current_open = current_price

        prices_pool.append(current_price)
        current_high = max(prices_pool)
        current_low = min(prices_pool)

        # collect results
        open_.append(current_open)
        high.append(current_high)
        low.append(current_low)
        close.append(current_price)
        candle_ids.append(candle)

        # update height if not accomplished, else is fixed
        if np.abs(height) < min_height:
            height += delta

        height_bool = np.abs(height) >= min_height
        bull_reversal = (current_high - current_price) > min_reversal
        bear_reversal = (current_price - current_low) > min_reversal

        if (height_bool and height > 0 and bull_reversal) or (height_bool and height < 0 and bear_reversal):
            # new candle setup
            prev_candle = candle
            candle += 1
            height = 0
            prices_pool = []

        # reversal_control.append((max(prices_pool, default=current_open) - current_price, min(prices_pool, default=current_open) -
        # current_price))

    candles = pd.Series(data=candle_ids, index=trades.index, name='Candle')
    highs = pd.Series(data=high, index=trades.index, name='High')
    lows = pd.Series(data=low, index=trades.index, name='Low')
    opens = pd.Series(data=open_, index=trades.index, name='Open')
    closes = pd.Series(data=close, index=trades.index, name='Close')
    # control = pd.Series(data=reversal_control, index=trades.index, name='control')

    # data = [candles, opens, highs, lows, closes, control]
    # data = [candles*10**-decimal_positions, opens*10**-decimal_positions, highs*10**-decimal_positions, lows*10**-decimal_positions,
    #         closes*10**-decimal_positions, trades['Quantity'], trades['Timestamp']]
    data = [candles, opens, highs, lows, closes, trades['Quantity'], trades['Timestamp']]

    ret = pd.concat(data, axis=1, keys=[s.name for s in data])

    klines = ret.groupby(['Candle']).agg({'Open': 'first', 'High': 'last', 'Low': 'last', 'Close': 'last', 'Quantity': 'sum',
                                          'Timestamp': 'first'})

    date_index = klines['Timestamp'].apply(convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
    klines.set_index(date_index, inplace=True)

    repair_decimals = float(10 ** -decimal_positions)
    klines['High'] *= repair_decimals
    klines['Low'] *= repair_decimals
    klines['Open'] *= repair_decimals
    klines['Close'] *= repair_decimals

    return klines


# def repeat_prices_by_quantity(data: pd.DataFrame, epsilon_quantity: float, price_col="Price", qty_col='Quantity') -> np.ndarray:
#     repeated_prices = []
#     for price, quantity in data[[price_col, qty_col]].values:
#         repeat_count = int(-(quantity // -epsilon_quantity))  # ceil division
#         repeated_prices.extend([price] * repeat_count)
#
#     return np.array(repeated_prices).reshape(-1, 1)

# def repeat_prices_by_quantity(data: pd.DataFrame, epsilon_quantity: float, price_col="Price", qty_col='Quantity'):
#     for price, quantity in data[[price_col, qty_col]].itertuples(index=False):
#         repeat_count = int(-(quantity // -epsilon_quantity))  # ceil division
#         for _ in range(repeat_count):
#             yield price


def repeat_prices_by_quantity(data: pd.DataFrame, epsilon_quantity: float, price_col="Price", qty_col='Quantity') -> np.ndarray:
    """
    Repeat prices by quantity to use in K-means clustering.

    :param pd.DataFrame data: A pandas DataFrame with trades or klines, containing a 'Price', 'Quantity' columns and a 'Buyer was maker' column,
        if trades passed, else "Close", "Volume" and "Taker buy base volume"
    :param float epsilon_quantity: The epsilon quantity to use for repeating prices.
    :param str price_col: The name of the column containing price data. Default is 'Price'.
    :param str qty_col: The name of the column containing quantity data. Default is 'Quantity'.
    :return np.ndarray: A numpy array with the prices repeated by quantity.
    """
    quantities = np.ceil(data[qty_col].values / epsilon_quantity).astype(int)
    repeated_prices = np.repeat(data[price_col].values, quantities)
    return repeated_prices.reshape(-1, 1)


def kmeans_custom_init(data: np.ndarray, max_clusters: int):
    """
    Generate initial centroids for K-means clustering with equally spaced values between min and max values in data.
    :param data: A data array.
    :param max_clusters: Max clusters expected.
    :return: A numpy array with initial centroids.
    """

    min_value = np.min(data)
    max_value = np.max(data)
    # Genera un array equidistante entre min_price y max_price con longitud max_clusters
    initial_centroids = np.linspace(min_value, max_value, max_clusters).reshape(-1, 1)
    return initial_centroids


def find_optimal_clusters(KMeans_lib, data: np.ndarray, max_clusters: int, quiet: bool = False,
                          initial_centroids: list or np.ndarray = None) -> int:
    """
    Find the optimal quantity of centroids for support and resistance methods using the elbow method.

    :param KMeans_lib: A KMeans library to use.
    :param data: A numpy array with the data to analyze.
    :param max_clusters: Maximum number of clusters to consider.
    :param bool quiet: If true, do not print progress bar.
    :param list initial_centroids: Initial centroids to use optionally for faster results.
    :return: The optimal number of clusters (centroids) as an integer.
    """

    inertia = []
    if quiet:
        pbar = range(1, max_clusters + 1)
    else:
        pbar = tqdm(range(1, max_clusters + 1))
    for n_clusters in pbar:
        kmeans = KMeans_lib(n_clusters=n_clusters, n_init=10, random_state=0, init=initial_centroids).fit(data)
        inertia.append(kmeans.inertia_)
    return np.argmin(np.gradient(np.gradient(inertia))) + 1


def support_resistance_levels(df: pd.DataFrame, max_clusters: int = 5, by_quantity: float = None, by_klines=True, quiet=False,
                              optimize_clusters_qty: bool = False) -> Tuple:
    """
    Calculate support and resistance levels for a given set of trades using K-means clustering.

    .. image:: images/indicators/support_resistance.png
           :width: 1000

    :param df: A pandas DataFrame with trades or klines, containing a 'Price', 'Quantity' columns and a 'Buyer was maker' column,
     if trades passed, else "Close", "Volume" and "Taker buy base volume"
    :param max_clusters: Maximum number of clusters to consider for finding the optimal number of centroids. Default: 5.
    :param float by_quantity: Count each price as many times the quantity contains a float of a the passed amount.
        Example: If a price 0.001 has a quantity of 100 and by_quantity is 0.1, quantity/by_quantity = 100/0.1 = 1000, then this prices
        is taken into account 1000 times instead of 1.
    :param bool by_klines: If true, use market profile from klines.
    :param bool quiet: If true, do not print progress bar.
    :param bool optimize_clusters_qty: If true, find the optimal number of clusters to use for calculating support and resistance levels.
    :return: A tuple containing two lists: the first list contains the support levels, and the second list contains
        the resistance levels. Both lists contain float values.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print(f"Please install sklearn: `pip install -U scikit-learn` to use Clustering")
        return [], []
    # copy data to avoid side effects
    df_ = df.copy(deep=True)

    # function core starts here
    if not by_klines:
        buy_data = df_.loc[df_['Buyer was maker'] == False]
        sell_data = df_.loc[df_['Buyer was maker'] == True]
        initial_centroids = kmeans_custom_init(data=df_['Price'].values, max_clusters=max_clusters)

    else:
        profile = market_profile_from_klines_melt(df=df_).reset_index()
        buy_data = profile.loc[profile['Is_Maker'] == True]
        sell_data = profile.loc[profile['Is_Maker'] == False]
        initial_centroids = kmeans_custom_init(data=df_['Close'].values, max_clusters=max_clusters)

    if by_quantity and not by_klines:
        buy_prices = repeat_prices_by_quantity(data=buy_data, epsilon_quantity=by_quantity, price_col="Price", qty_col="Quantity")
        sell_prices = repeat_prices_by_quantity(data=sell_data, epsilon_quantity=by_quantity, price_col="Price", qty_col="Quantity")
    elif by_quantity and by_klines:  # asume data came from Market Profile
        buy_prices = repeat_prices_by_quantity(data=buy_data, epsilon_quantity=by_quantity, price_col="Market_Profile", qty_col="Volume")
        sell_prices = repeat_prices_by_quantity(data=sell_data, epsilon_quantity=by_quantity, price_col="Market_Profile", qty_col="Volume")
    else:
        buy_prices = buy_data['Price'].values.reshape(-1, 1)
        sell_prices = sell_data['Price'].values.reshape(-1, 1)

    if len(buy_prices) == 0 and len(sell_prices) == 0:
        print(f"There is no trade data to calculate support and resistance levels: {len(buy_prices)} buys and {len(sell_prices)} sells.")
        return [], []

    if not quiet:
        print("Clustering data...")

    if optimize_clusters_qty:
        optimal_buy_clusters = find_optimal_clusters(KMeans_lib=KMeans, data=buy_prices, max_clusters=max_clusters, quiet=quiet,
                                                     initial_centroids=initial_centroids)
        optimal_sell_clusters = find_optimal_clusters(KMeans_lib=KMeans, data=sell_prices, max_clusters=max_clusters, quiet=quiet,
                                                      initial_centroids=initial_centroids)
        if not quiet:
            print(f"Found {optimal_buy_clusters} support levels from buys and {optimal_sell_clusters} resistance levels from sells.")
    else:
        optimal_buy_clusters = max_clusters
        optimal_sell_clusters = max_clusters

    kmeans_buy = KMeans(n_clusters=optimal_buy_clusters, n_init=1, init=initial_centroids).fit(buy_prices)
    kmeans_sell = KMeans(n_clusters=optimal_sell_clusters, n_init=1, init=initial_centroids).fit(sell_prices)

    support_levels = np.sort(kmeans_buy.cluster_centers_, axis=0)
    resistance_levels = np.sort(kmeans_sell.cluster_centers_, axis=0)

    return support_levels.flatten().tolist(), resistance_levels.flatten().tolist()


def support_resistance_levels_merged(df: pd.DataFrame,
                                     by_klines: bool,
                                     max_clusters: int = 5,
                                     by_quantity: float = None,
                                     optimize_clusters_qty: bool = False,
                                     quiet: bool = False):
    """
    Calculate support and resistance levels merged for a given set of trades using K-means clustering.
    :param df: A pandas DataFrame with trades or klines, containing a 'Price', 'Quantity' columns or "Close", "Volume".
    :param by_klines: If true, use klines.
    :param max_clusters: Quantity of clusters to consider initially. Default: 5.
    :param by_quantity: If true, use quantity to repeat prices. It gives more importance to prices with more quantity.
    :param optimize_clusters_qty: If true, find the optimal number of clusters to use for calculating support and resistance levels.
    :param quiet: If true, do not print progress bar.
    :return: A list containing the support and resistance levels merged. It would be just levels.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print(f"Please install sklearn: `pip install -U scikit-learn` to use Clustering")
        return [], []
    # copy data to avoid side effects
    df_ = df.copy(deep=True)

    # function core starts here
    if not by_klines:
        initial_centroids = kmeans_custom_init(data=df_['Price'].values, max_clusters=max_clusters)
    else:
        initial_centroids = kmeans_custom_init(data=df_['Close'].values, max_clusters=max_clusters)

    if by_quantity and not by_klines:
        repeated_prices = repeat_prices_by_quantity(data=df_, epsilon_quantity=by_quantity, price_col="Price", qty_col="Quantity")
    elif by_quantity and by_klines:
        repeated_prices = repeat_prices_by_quantity(data=df_, epsilon_quantity=by_quantity, price_col="Close", qty_col="Volume")
    else:
        repeated_prices = df_['Price'].values.reshape(-1, 1)

    if len(repeated_prices) == 0:
        print(f"There is no data to calculate support and resistance merged levels: {len(repeated_prices)}")
        return [], []
    if not quiet:
        print("Clustering data...")

    if optimize_clusters_qty:
        optimal_clusters = find_optimal_clusters(KMeans_lib=KMeans, data=repeated_prices, max_clusters=max_clusters, quiet=quiet,
                                                 initial_centroids=initial_centroids)
        if not quiet:
            print(f"Found {optimal_clusters} levels.")
    else:
        optimal_clusters = max_clusters

    kmeans_result = KMeans(n_clusters=optimal_clusters, n_init=1, init=initial_centroids).fit(repeated_prices)
    levels = np.sort(kmeans_result.cluster_centers_, axis=0)
    return levels.flatten().tolist()


def repeat_timestamps_by_quantity(df: pd.DataFrame, epsilon_quantity: float, buy_maker: bool = None, buy_taker: bool = None) -> np.ndarray:
    """
    Repeat timestamps by quantity to give more importance to prices with more quantity. It detects if data is from trades or klines by
    column names.

    :param df: A pandas DataFrame with trades or klines, containing a 'Price', 'Quantity' columns or "Close", "Volume" respectively.
    :param epsilon_quantity: Quantity to repeat timestamps by.
    :param buy_maker: If true, use maker side volume.
    :param buy_taker: If true, use taker side volume.
    :return:
    """
    data = df.copy()
    repeated_timestamps = []
    if 'Close' in data.columns:
        if buy_maker:
            data['Maker buy base volume'] = data['Volume'] - data['Taker buy base volume']
            col = 'Maker buy base volume'
        elif buy_taker:
            col = 'Taker buy base volume'
        else:
            col = 'Volume'

        for price, quantity, timestamp in data[['Close', col, 'Open timestamp']].values:
            repeat_count = int(-(quantity // -epsilon_quantity))  # ceil division
            repeated_timestamps.extend([timestamp] * repeat_count)
    else:
        if buy_maker:
            data_filtered = data.loc[data['Buyer was maker'] == True]
        elif buy_taker:
            data_filtered = data.loc[data['Buyer was maker'] == False]
        else:
            data_filtered = data
        for price, quantity, timestamp in data_filtered[['Price', 'Quantity', 'Timestamp']].values:
            repeat_count = int(-(quantity // -epsilon_quantity))  # ceil division
            repeated_timestamps.extend([timestamp] * repeat_count)
    return np.array(repeated_timestamps).reshape(-1, 1)


def time_active_zones(df: pd.DataFrame, max_clusters: int = 5, simple: bool = True, by_quantity: float = True, quiet=False,
                      optimize_clusters_qty: bool = False) -> Tuple:
    """
    Calculate support and resistance levels timestamp centroids for a given set of trades using K-means clustering.


    :param df: A pandas DataFrame with trades or klines, containing a 'Price', 'Quantity' columns and a 'Buyer was maker' column,
     if trades passed, else "Close", "Volume" and "Taker buy base volume"
    :param max_clusters: Maximum number of clusters to consider for finding the optimal number of centroids. Default: 5.
    :param bool simple: If true, use all trades to calculate time activity clusters.
    :param float by_quantity: Count each price as many times the quantity contains a float of a the passed amount.
        Example: If a price 0.001 has a quantity of 100 and by_quantity is 0.1, quantity/by_quantity = 100/0.1 = 1000, then this prices
        is taken into account 1000 times instead of 1.
    :param bool quiet: If true, do not print progress bar.
    :param bool optimize_clusters_qty: If true, find the optimal number of clusters to use for calculating support and resistance levels.
    :return: A tuple containing two lists: the first list contains the support levels, and the second list contains
        the resistance levels. Both lists contain float values.

    .. image:: images/indicators/time_action.png
           :width: 1000

    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print(f"Please install sklearn: `pip install -U scikit-learn` to use Clustering")
        return [], []

    # copy data to avoid side effects
    df_, my_timestamps, buy_timestamps, sell_timestamps = df.copy(deep=True), [], [], []

    # function core starts here
    if not 'Close' in df_.columns:
        initial_centroids = kmeans_custom_init(data=df_['Timestamp'].values, max_clusters=max_clusters)
        if not simple:
            buy_data = df_.loc[df_['Buyer was maker'] == False]
            sell_data = df_.loc[df_['Buyer was maker'] == True]
            buy_timestamps = repeat_timestamps_by_quantity(df=buy_data, epsilon_quantity=by_quantity, buy_maker=False, buy_taker=True)
            sell_timestamps = repeat_timestamps_by_quantity(df=sell_data, epsilon_quantity=by_quantity, buy_maker=True, buy_taker=False)
        else:
            my_timestamps = repeat_timestamps_by_quantity(df=df_, epsilon_quantity=by_quantity)  # tomara todo el quantity
    else:
        initial_centroids = kmeans_custom_init(data=df_['Open timestamp'].values, max_clusters=max_clusters)
        assert by_quantity, "If simple is true, by_quantity must be true too because KMEANS from evenly spaced klines has no sense."
        if not simple:
            buy_timestamps = repeat_timestamps_by_quantity(df=df_, epsilon_quantity=by_quantity, buy_maker=False, buy_taker=True)
            sell_timestamps = repeat_timestamps_by_quantity(df=df_, epsilon_quantity=by_quantity, buy_maker=True, buy_taker=False)
        else:
            my_timestamps = repeat_timestamps_by_quantity(df=df_, epsilon_quantity=by_quantity)  # tomara todo el volumen

    if simple:
        if len(my_timestamps) == 0:
            print(f"There is not enough trade data to calculate time activity clusters: {len(my_timestamps)} timestamps.")
            return [], []
    else:
        if len(buy_timestamps) == 0 and len(sell_timestamps) == 0:
            print(f"There is not enough trade data to calculate time activity clusters: {len(buy_timestamps)} buys and "
                  f"{len(sell_timestamps)} sells.")
            return [], []

    if not quiet:
        print("Clustering data...")
    optimal_buy_clusters, optimal_sell_clusters, optimal_clusters = max_clusters, max_clusters, max_clusters
    if optimize_clusters_qty and not simple:
        optimal_buy_clusters = find_optimal_clusters(KMeans_lib=KMeans, data=buy_timestamps, max_clusters=max_clusters, quiet=quiet,
                                                     initial_centroids=initial_centroids)
        optimal_sell_clusters = find_optimal_clusters(KMeans_lib=KMeans, data=sell_timestamps, max_clusters=max_clusters, quiet=quiet,
                                                      initial_centroids=initial_centroids)
        print(f"Found {optimal_buy_clusters} clusters from buys timestamps and {optimal_sell_clusters} clusters from sells timestamps.")
    elif optimize_clusters_qty and simple:
        optimal_clusters = find_optimal_clusters(KMeans_lib=KMeans, data=my_timestamps, max_clusters=max_clusters, quiet=quiet,
                                                 initial_centroids=initial_centroids)
        print(f"Found {optimal_clusters} clusters from timestamps.")

    n_init = 10
    if initial_centroids is not None:
        n_init = 1

    if not simple:
        kmeans_buy = KMeans(n_clusters=optimal_buy_clusters, n_init=n_init, init=initial_centroids).fit(buy_timestamps)
        kmeans_sell = KMeans(n_clusters=optimal_sell_clusters, n_init=n_init, init=initial_centroids).fit(sell_timestamps)
        support_levels = np.sort(kmeans_buy.cluster_centers_, axis=0)
        resistance_levels = np.sort(kmeans_sell.cluster_centers_, axis=0)
        return support_levels.flatten().tolist(), resistance_levels.flatten().tolist()

    else:
        kmeans_result = KMeans(n_clusters=optimal_clusters, n_init=n_init, init=initial_centroids).fit(my_timestamps)
        support_levels = np.sort(kmeans_result.cluster_centers_, axis=0)
        return support_levels.flatten().tolist(), []


def market_profile_from_klines_melt(df: pd.DataFrame):
    """
    Calculate the market profile for a given OHLC data. The function calculates the average price for each candle
    (high + low + close) / 3, and then calculates the 'maker' and 'taker' volumes for each average price.

    :param df: A pandas DataFrame with the OHLC data. It should contain 'High', 'Low', 'Close', 'Volume', and
               'Taker buy base volume' columns.
    :return: A pandas DataFrame grouped by the average price ('Market_Profile') with the sum of 'Taker buy base volume'
             and 'Maker_Volume' for each average price.
    """
    df_ = df.copy(deep=True)
    df_['Market_Profile'] = (df_['High'] + df_['Low'] + df_['Close']) / 3
    df_['Maker buy base volume'] = df_['Volume'] - df_['Taker buy base volume']

    # Rename the existing 'Volume' column
    df_.rename(columns={'Volume': 'Total_Volume'}, inplace=True)

    # Melt the dataframe to unpivot the volume columns
    df_melt = df_.melt(id_vars='Market_Profile', value_vars=['Taker buy base volume',
                                                             'Maker buy base volume'], var_name='Is_Maker', value_name='Volume')

    # Convert the 'Is_Maker' column to boolean
    df_melt['Is_Maker'] = df_melt['Is_Maker'] == 'Maker buy base volume'

    # Group by 'Market_Profile' and 'Is_Maker' and sum the volumes
    df_grouped = df_melt.groupby(['Market_Profile', 'Is_Maker']).agg({'Volume': 'sum'})

    return df_grouped.sort_index(level='Market_Profile')


def taker_maker_profile_from_klines_melt(df: pd.DataFrame):
    """
    Calculate the ratio of taker and maker volume for a given OHLC data.

    :param df: A pandas DataFrame with the OHLC data. It should contain 'High', 'Low', 'Close', 'Volume', and
               'Taker buy base volume' columns.
    :return: A pandas DataFrame grouped by the average price ('Market_Profile') with the sum of 'Taker buy base volume'
             and 'Maker_Volume' for each average price.
    """
    df_ = df.copy(deep=True)

    df_['Maker buy base volume'] = df_['Volume'] - df_['Taker buy base volume']
    df_['Ratio_Profile'] = df_['Taker buy base volume'] / df_["Volume"]

    # Rename the existing 'Volume' column
    df_.rename(columns={'Volume': 'Total_Volume'}, inplace=True)

    # Melt the dataframe to unpivot the volume columns
    df_melt = df_.melt(id_vars='Ratio_Profile', value_vars=['Taker buy base volume',
                                                            'Maker buy base volume'], var_name='Is_Maker', value_name='Volume')

    # Convert the 'Is_Maker' column to boolean
    df_melt['Is_Maker'] = df_melt['Is_Maker'] == 'Maker buy base volume'

    # Group by 'Market_Profile' and 'Is_Maker' and sum the volumes
    df_grouped = df_melt.groupby(['Ratio_Profile', 'Is_Maker']).agg({'Volume': 'sum'})

    return df_grouped.sort_index(level='Ratio_Profile')


def market_profile_from_klines_grouped(df: pd.DataFrame, num_bins: int = 100) -> pd.DataFrame:
    """
    Calculate the market profile for a given OHLC data. The function calculates the average price for each candle
    (high + low + close) / 3, and then calculates the 'maker' and 'taker' volumes for each average price.

    :param df: A pandas DataFrame with the OHLC data. It should contain 'High', 'Low', 'Close', 'Volume', and
               'Taker buy base volume' columns.
    :param int num_bins: Number of bins to use for the market profile.
    :return: A pandas DataFrame grouped by the average price ('Market_Profile') with the sum of 'Taker buy base volume'
             and 'Maker_Volume' for each average price.
    """
    df_ = df.copy(deep=True)
    df_['Maker buy base volume'] = df_['Volume'] - df_['Taker buy base volume']
    df_['Market_Profile'] = (df_['High'] + df_['Low'] + df_['Close']) / 3
    df_['Price_Bin'] = pd.cut(df_['Market_Profile'], bins=num_bins)
    volume_by_price_bin = df_.groupby('Price_Bin').agg({'Taker buy base volume': 'sum', 'Maker buy base volume': 'sum'})
    # Convertir el índice a un IntervalIndex
    volume_by_price_bin.index = pd.IntervalIndex(volume_by_price_bin.index)
    # Ordenar por el límite inferior de cada intervalo
    volume_by_price_bin = volume_by_price_bin.sort_index(key=lambda x: x.left)
    volume_by_price_bin.index.name += f"_{df_.index.name}_Klines"
    return volume_by_price_bin


def market_profile_from_trades_grouped(df: pd.DataFrame, num_bins: int = 100) -> pd.DataFrame:
    """
    Calculate the market profile for a given trades data. The function calculates the average price for each trade
    and then calculates the 'maker' and 'taker' volumes for each average price.

    :param df: A pandas DataFrame with the trades data. It should contain 'Price', 'Quantity', 'Buyer was maker' columns.
    :param num_bins: The number of bins to use for the market profile.
    :return: A pandas DataFrame grouped by the average price ('Price_Bin') with the sum of 'Taker buy base volume'
             and 'Maker buy base volume' for each average price.
    """
    df_ = df.copy(deep=True)
    df_['Taker buy base volume'] = df_['Quantity'].where(~df_['Buyer was maker'], 0)
    df_['Maker buy base volume'] = df_['Quantity'].where(df_['Buyer was maker'], 0)

    df_['Price_Bin'] = pd.cut(df_['Price'], bins=num_bins)
    volume_by_price_bin = df_.groupby('Price_Bin').agg({'Taker buy base volume': 'sum', 'Maker buy base volume': 'sum'})
    # Convert the index to an IntervalIndex
    volume_by_price_bin.index = pd.IntervalIndex(volume_by_price_bin.index)
    # Sort by the lower bound of each interval
    volume_by_price_bin = volume_by_price_bin.sort_index(key=lambda x: x.left)
    volume_by_price_bin.index.name += f"_{df_.index.name}_Trades"
    return volume_by_price_bin
