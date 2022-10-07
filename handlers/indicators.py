import numpy as np
import pandas as pd


def ichimoku(df: pd.DataFrame,
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

    :param pd.DataFrame df: A BinPan Symbol dataframe.
    :param int tenkan: The short period. It's the half sum of max and min price in the window. Default: 9
    :param int kijun: The long period. It's the half sum of max and min price in the window. Default: 26
    :param int chikou_span: Close of the next 26 bars. Util when spoting what happened with other ichimoku lines and what happened
      before Default: 26.
    :param senkou_cloud_base: Period to obtain kumo cloud base line. Default is 52.
    :param str suffix: A decorative suffix for the name of the column created.
    :return pd.DataFrame: A pandas dataframe with columns as indicator lines.
    """

    high = df['High']
    low = df['Low']
    close = df['Close']

    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).max()) / 2
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).max()) / 2

    chikou_span_serie = close.shift(-chikou_span)

    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b_arr = ((high.rolling(window=senkou_cloud_base).max() + low.rolling(window=senkou_cloud_base).max()) / 2).to_numpy()

    # desplazamiento de 26 barras, la corriente es la 0
    span = np.array([np.nan] * chikou_span)
    freq = pd.infer_freq(df.index)
    kumo_index_ahead = df.index.shift(chikou_span, freq=freq)
    kumo_index = np.concatenate([df.index[:chikou_span], kumo_index_ahead], axis=None)

    senkou_span_b = pd.Series(data=np.concatenate([span, senkou_span_b_arr]), index=kumo_index)

    # just one possible step value 1m, or 5m or any step but one step
    # assert len(senkou_span_b.index.to_series().diff().value_counts()) == 1

    ret = pd.DataFrame([tenkan_sen, kijun_sen, chikou_span_serie, senkou_span_a, senkou_span_b]).T

    if suffix:
        suffix = '_' + suffix

    ret.columns = [f"Ichimoku_tenkan_{tenkan}"+suffix,
                   f"Ichimoku_kijun_{kijun}"+suffix,
                   f"Ichimoku_chikou_span{chikou_span}"+suffix,
                   f"Ichimoku_cloud_{tenkan}_{kijun}"+suffix,
                   f"Ichimoku_cloud_{senkou_cloud_base}"+suffix]
    return ret
