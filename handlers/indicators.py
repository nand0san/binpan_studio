"""

BinPan own indicators and utils.

"""
import pandas as pd
from tqdm import tqdm
import pytz
import numpy as np

from .time_helper import convert_milliseconds_to_time_zone_datetime
from .time_helper import pandas_freq_tick_interval
from .tags import is_alternating


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
    :param int max_period: Default is len of df. This method will check from 2 to the max period value to find a puer alternating max to mins.
    :param str suffix: A decorative suffix for the name of the column created.
    :return pd.DataFrame: A dataframe with two columns, one with 1 or -1 for local max or local min to tag, and other with price values for
     that points.

    .. image:: images/indicators/fractal_w.png
        :width: 1000
    """
    fractal = None
    if not max_period:
        max_period = len(df)
    for i in tqdm(range(2, max_period)):
        fractal = fractal_w_indicator(data=df, period=i, suffix=suffix, fill_with_zero=True)
        max_min = fractal[f"Fractal_W_{i}"].fillna(0)
        if is_alternating(lst=max_min.tolist()):
            break
        else:
            fractal = None
    return fractal


def fractal_trend_indicator(df: pd.DataFrame, period: int = None, fractal: pd.DataFrame = None, suffix: str = "") -> tuple or None:
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

            from binpan import binpan

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
    Kaufman's Efficiency Ratio based in: https://stackoverflow.com/questions/36980238/calculating-kaufmans-efficiency-ratio-in-python-with-pandas

    :param pd.Series close: Close prices serie.
    :param int window: Window to check indicator.
    :return pd.Series: Results.
    """
    direction = close.diff(window).abs()
    # noinspection PyUnresolvedReferences
    volatility = pd.rolling_sum(close.diff().abs(), window)
    return direction / volatility


def fractal_w_indicator(data: pd.DataFrame, period=2, merged: bool = True, suffix: str = '', fill_with_zero: bool = None, ) -> pd.DataFrame:
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
    :param bool fill_with_zero: If true fills nans with zeros. Its better to plot with binpan.
    :return pd.DataFrame: A dataframe with two columns, one with 1 or -1 for local max or local min to tag, and other with price values for
     that points.

    .. image:: images/indicators/fractal_w.png
        :width: 1000
    """
    window = 2 * period + 1  # default 5

    mins = data['Low'].rolling(window, center=True).apply(lambda x: x[period] == min(x), raw=True)
    maxs = data['High'].rolling(window, center=True).apply(lambda x: x[period] == max(x), raw=True)

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

    values = pd.Series(np.nan, index=data.index)
    values.loc[min_idx] = data['Low']
    values.loc[max_idx] = data['High']
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
    :param int threshold: Percentil to show most traded levels. Default is 90.
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
    :param int decimal_positions: Because this function uses integer numbers for prices, is needed to convert prices. Just steps are relevant.
    :param str time_zone: A time zone like "Europe/Madrid".
    :param int min_height: Minimum candles height in pips.
    :param int min_reversal: Maximum reversal to close the candles
    :return pd.DataFrame: A serie with the resulting candles number sequence.

     Example:
        .. code-block:: python

           from binpan import binpan

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

        # reversal_control.append((max(prices_pool, default=current_open) - current_price, min(prices_pool, default=current_open) - current_price))

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


def repeat_prices_by_quantity(data: pd.DataFrame, epsilon_quantity: float, price_col="Prices", qty_col='Quantity') -> np.ndarray:
    repeated_prices = []
    for price, quantity in data[[price_col, qty_col]].values:
        repeat_count = int(-(quantity // -epsilon_quantity))  # ceil division
        repeated_prices.extend([price] * repeat_count)

    return np.array(repeated_prices).reshape(-1, 1)


def support_resistance_levels(data: pd.DataFrame, max_clusters: int = 10, by_quantity: float = None, by_klines=True) -> tuple:
    """
    Calculate support and resistance levels for a given set of trades using K-means clustering.

    .. image:: images/indicators/support_resistance.png
           :width: 1000

    :param data: A pandas DataFrame with trade data or klines, containing a 'Price', 'Quantity' columns and a 'Buyer was maker' column.
    :param max_clusters: Maximum number of clusters to consider for finding the optimal number of centroids.
    :param float by_quantity: Count each price as many times the quantity contains a float of a the passed amount.
        Example: If a price 0.001 has a quantity of 100 and by_quantity is 0.1, quantity/by_quantity = 100/0.1 = 1000, then this prices
        is taken into account 1000 times instead of 1.
    :param bool by_klines: If true, use market profile from klines.
    :return: A tuple containing two lists: the first list contains the support levels, and the second list contains
        the resistance levels. Both lists contain float values.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print(f"Please install sklearn: `pip install -U scikit-learn` to use Clustering")
        return [], []
    data = data.copy(deep=True)

    def find_optimal_clusters(data: np.ndarray, max_clusters: int) -> int:
        """
        Find the optimal quantity of centroids for support and resistance methods using the elbow method.

        :param data: A numpy array with the data to analyze.
        :param max_clusters: Maximum number of clusters to consider.
        :return: The optimal number of clusters (centroids) as an integer.
        """

        inertia = []
        for n_clusters in tqdm(range(1, max_clusters + 1)):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(data)
            inertia.append(kmeans.inertia_)
        return np.argmin(np.gradient(np.gradient(inertia))) + 1

    if not by_klines:
        buy_data = data.loc[data['Buyer was maker'] == False]
        sell_data = data.loc[data['Buyer was maker'] == True]
    else:
        profile = market_profile_from_klines_melt(data=data).reset_index()
        buy_data = profile.loc[profile['Is_Maker'] == True]
        sell_data = profile.loc[profile['Is_Maker'] == False]

    if by_quantity and not by_klines:
        buy_prices = repeat_prices_by_quantity(data=buy_data, epsilon_quantity=by_quantity)
        sell_prices = repeat_prices_by_quantity(data=sell_data, epsilon_quantity=by_quantity)
    elif by_quantity and by_klines:
        buy_prices = repeat_prices_by_quantity(data=buy_data, epsilon_quantity=by_quantity, price_col="Market_Profile", qty_col="Volume")
        sell_prices = repeat_prices_by_quantity(data=sell_data, epsilon_quantity=by_quantity, price_col="Market_Profile", qty_col="Volume")
    else:
        buy_prices = buy_data['Price'].values.reshape(-1, 1)
        sell_prices = sell_data['Price'].values.reshape(-1, 1)

    if len(buy_prices) == 0 and len(sell_prices) == 0:
        print("There is not enough trade data to calculate support and resistance levels.")
        return [], []

    print("Clustering data...")
    optimal_buy_clusters = find_optimal_clusters(buy_prices, max_clusters)
    optimal_sell_clusters = find_optimal_clusters(sell_prices, max_clusters)

    print(f"Found {optimal_buy_clusters} support levels from buys and {optimal_sell_clusters} resistance levels from sells.")

    kmeans_buy = KMeans(n_clusters=optimal_buy_clusters, n_init=10).fit(buy_prices)
    kmeans_sell = KMeans(n_clusters=optimal_sell_clusters, n_init=10).fit(sell_prices)

    support_levels = np.sort(kmeans_buy.cluster_centers_, axis=0)
    resistance_levels = np.sort(kmeans_sell.cluster_centers_, axis=0)

    return support_levels.flatten().tolist(), resistance_levels.flatten().tolist()


def repeat_timestamps_by_quantity(data: pd.DataFrame, epsilon_quantity: float) -> np.ndarray:
    repeated_prices = []
    for price, quantity, timestamp in data[['Price', 'Quantity', 'Timestamp']].values:
        repeat_count = int(-(quantity // -epsilon_quantity))  # ceil division
        repeated_prices.extend([timestamp] * repeat_count)
    return np.array(repeated_prices).reshape(-1, 1)


def time_active_zones(data: pd.DataFrame, max_clusters: int = 10, by_quantity: float = None) -> tuple:
    """
    Calculate active points in time by clustering timestamps of trades.

    .. image:: images/indicators/support_resistance.png
           :width: 1000

    :param data: A pandas DataFrame with trade data, containing a 'Price', 'Timestamp', 'Quantity' columns and a 'Buyer was maker' column.
    :param max_clusters: Maximum number of clusters to consider for finding the optimal number of centroids.
    :param float by_quantity: Count each price as many times the quantity contains a float of a the passed amount.
        Example: If a price 0.001 has a quantity of 100 and by_quantity is 0.1, quantity/by_quantity = 100/0.1 = 1000, then this prices
        is taken into account 1000 times instead of 1.
    :return: A tuple containing the most traded centroides timestamps..
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print(f"Please install sklearn: `pip install -U scikit-learn` to use Clustering")
        return [], []
    data = data.copy(deep=True)

    def find_optimal_clusters(data: np.ndarray, max_clusters: int) -> int:
        """
        Find the optimal quantity of centroids for support and resistance methods using the elbow method.

        :param data: A numpy array with the data to analyze.
        :param max_clusters: Maximum number of clusters to consider.
        :return: The optimal number of clusters (centroids) as an integer.
        """

        inertia = []
        for n_clusters in tqdm(range(1, max_clusters + 1)):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(data)
            inertia.append(kmeans.inertia_)
        return np.argmin(np.gradient(np.gradient(inertia))) + 1

    buy_data = data.loc[data['Buyer was maker'] == False]
    sell_data = data.loc[data['Buyer was maker'] == True]
    if by_quantity:
        buy_timestamps = repeat_timestamps_by_quantity(data=buy_data, epsilon_quantity=by_quantity)
        sell_timestamps = repeat_timestamps_by_quantity(data=buy_data, epsilon_quantity=by_quantity)
    else:
        buy_timestamps = buy_data['Timestamp'].values.reshape(-1, 1)
        sell_timestamps = sell_data['Timestamp'].values.reshape(-1, 1)

    if len(buy_timestamps) == 0 and len(sell_timestamps) == 0:
        print("There is not enough trade data to calculate time activity clusters.")
        return [], []
    print("Clustering data...")
    optimal_buy_clusters = find_optimal_clusters(buy_timestamps, max_clusters)
    optimal_sell_clusters = find_optimal_clusters(sell_timestamps, max_clusters)

    print(f"Found {optimal_buy_clusters} clusters from buys and {optimal_sell_clusters} clusters from sells.")

    kmeans_buy = KMeans(n_clusters=optimal_buy_clusters, n_init=10).fit(buy_timestamps)
    kmeans_sell = KMeans(n_clusters=optimal_sell_clusters, n_init=10).fit(sell_timestamps)

    support_levels = np.sort(kmeans_buy.cluster_centers_, axis=0)
    resistance_levels = np.sort(kmeans_sell.cluster_centers_, axis=0)

    return support_levels.flatten().tolist(), resistance_levels.flatten().tolist()


# def market_profile(data: pd.DataFrame):
#     """
#     Calculate the market profile for a given OHLC data. The function calculates the average price for each candle
#     (high + low + close) / 3, and then calculates the 'maker' and 'taker' volumes for each average price.
#
#     :param data: A pandas DataFrame with the OHLC data. It should contain 'High', 'Low', 'Close', 'Volume', and
#                'Taker buy base volume' columns.
#     :return: A pandas DataFrame grouped by the average price ('Market_Profile') with the sum of 'Taker buy base volume'
#              and 'Maker_Volume' for each average price.
#     """
#     df = data.copy(deep=True)
#     df['Market_Profile'] = (df['High'] + df['Low'] + df['Close']) / 3
#     df['Maker buy base volume'] = df['Volume'] - df['Taker buy base volume']
#     df_grouped = df.groupby('Market_Profile').agg({'Taker buy base volume': 'sum', 'Maker buy base volume': 'sum'})
#     df_grouped['Volume'] = df_grouped['Taker buy base volume'] + df_grouped['Maker buy base volume']
#     return df_grouped.sort_index()

def market_profile_from_klines_melt(data: pd.DataFrame):
    """
    Calculate the market profile for a given OHLC data. The function calculates the average price for each candle
    (high + low + close) / 3, and then calculates the 'maker' and 'taker' volumes for each average price.

    :param data: A pandas DataFrame with the OHLC data. It should contain 'High', 'Low', 'Close', 'Volume', and
               'Taker buy base volume' columns.
    :return: A pandas DataFrame grouped by the average price ('Market_Profile') with the sum of 'Taker buy base volume'
             and 'Maker_Volume' for each average price.
    """
    df = data.copy(deep=True)
    df['Market_Profile'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Maker buy base volume'] = df['Volume'] - df['Taker buy base volume']

    # Rename the existing 'Volume' column
    df.rename(columns={'Volume': 'Total_Volume'}, inplace=True)

    # Melt the dataframe to unpivot the volume columns
    df_melt = df.melt(id_vars='Market_Profile', value_vars=['Taker buy base volume',
                                                            'Maker buy base volume'], var_name='Is_Maker', value_name='Volume')

    # Convert the 'Is_Maker' column to boolean
    df_melt['Is_Maker'] = df_melt['Is_Maker'] == 'Maker buy base volume'

    # Group by 'Market_Profile' and 'Is_Maker' and sum the volumes
    df_grouped = df_melt.groupby(['Market_Profile', 'Is_Maker']).agg({'Volume': 'sum'})

    return df_grouped.sort_index(level='Market_Profile')


def market_profile_from_klines_grouped(data: pd.DataFrame, num_bins: int = 100) -> pd.DataFrame:
    """
    Calculate the market profile for a given OHLC data. The function calculates the average price for each candle
    (high + low + close) / 3, and then calculates the 'maker' and 'taker' volumes for each average price.

    :param data: A pandas DataFrame with the OHLC data. It should contain 'High', 'Low', 'Close', 'Volume', and
               'Taker buy base volume' columns.
    :param int num_bins: Number of bins to use for the market profile.
    :return: A pandas DataFrame grouped by the average price ('Market_Profile') with the sum of 'Taker buy base volume'
             and 'Maker_Volume' for each average price.
    """
    df = data.copy(deep=True)
    df['Maker buy base volume'] = df['Volume'] - df['Taker buy base volume']
    df['Market_Profile'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Price_Bin'] = pd.cut(df['Market_Profile'], bins=num_bins)
    volume_by_price_bin = df.groupby('Price_Bin').agg({'Taker buy base volume': 'sum', 'Maker buy base volume': 'sum'})
    # Convertir el índice a un IntervalIndex
    volume_by_price_bin.index = pd.IntervalIndex(volume_by_price_bin.index)
    # Ordenar por el límite inferior de cada intervalo
    volume_by_price_bin = volume_by_price_bin.sort_index(key=lambda x: x.left)
    volume_by_price_bin.index.name += f"_{df.index.name}_Klines"
    return volume_by_price_bin


def market_profile_from_trades_grouped(data: pd.DataFrame, num_bins: int = 100) -> pd.DataFrame:
    """
    Calculate the market profile for a given trades data. The function calculates the average price for each trade
    and then calculates the 'maker' and 'taker' volumes for each average price.

    :param data: A pandas DataFrame with the trades data. It should contain 'Price', 'Quantity', 'Buyer was maker' columns.
    :param num_bins: The number of bins to use for the market profile.
    :return: A pandas DataFrame grouped by the average price ('Price_Bin') with the sum of 'Taker buy base volume'
             and 'Maker buy base volume' for each average price.
    """
    df = data.copy(deep=True)
    df['Taker buy base volume'] = df['Quantity'].where(~df['Buyer was maker'], 0)
    df['Maker buy base volume'] = df['Quantity'].where(df['Buyer was maker'], 0)

    df['Price_Bin'] = pd.cut(df['Price'], bins=num_bins)
    volume_by_price_bin = df.groupby('Price_Bin').agg({'Taker buy base volume': 'sum', 'Maker buy base volume': 'sum'})
    # Convert the index to an IntervalIndex
    volume_by_price_bin.index = pd.IntervalIndex(volume_by_price_bin.index)
    # Sort by the lower bound of each interval
    volume_by_price_bin = volume_by_price_bin.sort_index(key=lambda x: x.left)
    volume_by_price_bin.index.name += f"_{df.index.name}_Trades"
    return volume_by_price_bin
