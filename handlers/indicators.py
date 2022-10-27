"""

BinPan own indicators and utils.

"""
import numpy as np
import pandas as pd
from typing import Tuple

from handlers.time_helper import pandas_freq_tick_interval


##############
# INDICATORS #
##############

def ichimoku(data: pd.DataFrame,
             tenkan: int = 9,
             kijun: int = 26,
             chikou_span: int = 26,
             senkou_cloud_base: int = 52,
             suffix: str = ''
             ) -> pd.DataFrame:
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

            from binpan import binpan

            sym = binpan.Symbol(symbol='LUNCBUSD', tick_interval='1m', limit=500)
            sym.ichimoku()
            sym.plot()

       .. image:: images/indicators/ichimoku.png
        :width: 1000
        :alt: Candles with some indicators

    """
    df = data.copy(deep=True)

    tick_interval = df.index.name.split()[1]
    df = df.asfreq(pandas_freq_tick_interval[tick_interval])

    high = df['High']
    low = df['Low']
    close = df['Close']
    # freq = pd.infer_freq(df.index)

    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2

    chikou_span_serie = close.shift(periods=-chikou_span, freq='infer')

    arr_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_a = arr_a.shift(periods=chikou_span, freq='infer')

    arr_b = (high.rolling(window=senkou_cloud_base).max() + low.rolling(window=senkou_cloud_base).min()) / 2
    senkou_span_b = arr_b.shift(periods=chikou_span, freq='infer')

    ret = pd.DataFrame([tenkan_sen, kijun_sen, chikou_span_serie, senkou_span_a, senkou_span_b]).T

    if suffix:
        suffix = '_' + suffix

    ret.columns = [f"Ichimoku_tenkan_{tenkan}" + suffix,
                   f"Ichimoku_kijun_{kijun}" + suffix,
                   f"Ichimoku_chikou_span{chikou_span}" + suffix,
                   f"Ichimoku_cloud_{tenkan}_{kijun}" + suffix,
                   f"Ichimoku_cloud_{senkou_cloud_base}" + suffix]
    # return ret.dropna(how='all', inplace=True)
    return ret


def ker(close: pd.Series,
        window: int,
        ) -> pd.Series:
    """
    Kaufman's Efficiency Ratio based in: https://stackoverflow.com/questions/36980238/calculating-kaufmans-efficiency-ratio-in-python-with-pandas

    :param pd.Series close: Close prices serie.
    :param int window: Window to check indicator.
    :return pd.Series: Results.
    """
    direction = close.diff(window).abs()
    volatility = pd.rolling_sum(close.diff().abs(), window)
    return direction / volatility


def fractal_w(data: pd.DataFrame,
              period=2,
              merged: bool = True,
              suffix: str = '',
              ) -> pd.DataFrame:
    """
    The fractal indicator is based on a simple price pattern that is frequently seen in financial markets. Outside of trading, a fractal
    is a recurring geometric pattern that is repeated on all time frames. From this concept, the fractal indicator was devised.
    The indicator isolates potential turning points on a price chart. It then draws arrows to indicate the existence of a pattern.

    https://www.investopedia.com/terms/f/fractal.asp

    From: https://codereview.stackexchange.com/questions/259703/william-fractal-technical-indicator-implementation

    :param pd.DataFrame data: BinPan dataframe with High prices.
    :param int period: Default is 2. Count of neighbour candles to match max or min tags.
    :param bool merged: If True, values are merged into one pd.Serie. minimums overwrite maximums in case of coincidence.
    :param str suffix: A decorative suffix for the name of the column created.
    :return pd.Series: A serie with 1 or -1 for local max or local min to tag.
    """
    window = 2 * period + 1  # default 5

    mins = data['Low'].rolling(window, center=True).apply(lambda x: x[period] == min(x), raw=True)
    maxs = data['High'].rolling(window, center=True).apply(lambda x: x[period] == max(x), raw=True)

    mins = mins.replace({0: np.nan})
    maxs = maxs.replace({0: np.nan})
    maxs.loc[:] = maxs * -1

    # return mins, maxs
    if suffix:
        suffix = '_' + suffix

    mins.name = f"Fractal_W_Min_{period}" + suffix
    maxs.name = f"Fractal_W_Max_{period}" + suffix

    min_idx = mins.dropna().index
    max_idx = maxs.dropna().index

    values = pd.Series(np.nan, index=data.index)
    values.loc[min_idx] = data['Low']
    values.loc[max_idx] = data['High']
    values.name = f"Fractal_W_{period}_values" + suffix

    if not merged:
        return pd.DataFrame([mins, maxs, values]).T

    else:
        merged = mins
        merged.fillna(maxs, inplace=True)
        merged.name = f"Fractal_W_{period}" + suffix
        return pd.DataFrame([merged, values]).T


####################
# INDICATORS UTILS #
####################

def split_serie_by_position(serie: pd.Series,
                            splitter_serie: pd.Series,
                            fill_with_zeros: bool = True) -> pd.DataFrame:
    """
    Splits a serie by values of other serie in four series for plotting purposes. It means will get 4 series with different situations:

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


def zoom_cloud_indicators(plot_splitted_serie_couples: dict,
                          main_index: list,
                          start_idx: int,
                          end_idx: int) -> dict:
    """
    It zooms the cloud indicators in an index interval for a plot zoom.

    :param dict plot_splitted_serie_couples: Splitted indicators for cloud colored area plotting.
    :param list main_index: The BinPan general index for cutting.
    :param int start_idx: A index to cut.
    :param int end_idx: A index to cut.
    :return dict: All indicators cut.
    """
    try:
        assert start_idx < end_idx <= len(main_index)
    except AssertionError:
        raise Exception(f"BinPan Plot Error: Zoom index not valid. Not start={start_idx} < end={end_idx} < len={len(main_index)}")

    ret = {}
    my_start = main_index[start_idx]
    my_end = main_index[end_idx]
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
    It shifts a candle ahead by the window argument value (or backwards if negative)

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
