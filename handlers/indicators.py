import numpy as np
import pandas as pd


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
    :param int chikou_span: Close of the next 26 bars. Util when spoting what happened with other ichimoku lines and what happened
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
    # freq = pd.infer_freq(df.index)
    print(df)

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
    print(ret)

    if suffix:
        suffix = '_' + suffix

    ret.columns = [f"Ichimoku_tenkan_{tenkan}"+suffix,
                   f"Ichimoku_kijun_{kijun}"+suffix,
                   f"Ichimoku_chikou_span{chikou_span}"+suffix,
                   f"Ichimoku_cloud_{tenkan}_{kijun}"+suffix,
                   f"Ichimoku_cloud_{senkou_cloud_base}"+suffix]
    return ret


####################
# INDICATORS UTILS #
####################

def split_serie_by_position(serie: pd.Series,
                            splitter_serie: pd.Series,
                            fill_with_zeros: bool = True) -> pd.DataFrame:
    """
    Splits a serie by other serie values in four series for plotting purposes.

    :param pd.Series serie: A serie to classify in reference to other serie.
    :param pd.Series splitter_serie: A serie to split in two couple of series classified by position reference.
    :param bool fill_with_zeros: Fill nans with zeros for splitted lines like MACD to avoid artifacts in plots.
    :return tuple: A tuple with four series classified by upper position or lower position.
    """
    serie_up = serie.loc[serie.ge(splitter_serie)]
    splitter_up = splitter_serie.loc[serie_up.index]

    serie_down = serie.loc[serie.lt(splitter_serie)]
    splitter_down = splitter_serie.loc[serie_down.index]

    serie_up.name = serie_up.name + '_up'
    serie_down.name = serie_down.name + '_down'
    splitter_up.name = splitter_up.name + '_split_up'
    splitter_down.name = splitter_down.name + '_split_down'

    # return serie_up, serie_splitter_up, serie_down, serie_splitter_down
    # return serie_up, serie_down

    ret = pd.DataFrame([serie_up, splitter_up, serie_down, splitter_down]).T
    if fill_with_zeros:
        return ret.fillna(0)
    else:
        return ret
