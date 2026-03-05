"""
Indicators mixin for Symbol class.
"""
from __future__ import annotations

from time import time

import pandas as pd
import numpy as np

from handlers.exceptions import BinPanException
from handlers.indicators import (df_splitter, ichimoku, fractal_w_indicator,
                                 support_resistance_levels, market_profile_from_klines_grouped,
                                 alternating_fractal_indicator, fractal_trend_indicator,
                                 market_profile_from_trades_grouped, support_resistance_levels_merged, time_active_zones,
                                 atr as atr_indicator, supertrend as supertrend_indicator,
                                 macd as macd_indicator, stoch_rsi as stoch_rsi_indicator,
                                 obv as obv_indicator, ad as ad_indicator, vwap as vwap_indicator,
                                 cci as cci_indicator, eom as eom_indicator, roc as roc_indicator,
                                 bbands as bbands_indicator, stoch as stoch_indicator)
from handlers.logs import LogManager
from handlers.time_helper import (get_dataframe_time_index_ranges, remove_initial_included_ranges,
                                  pandas_freq_tick_interval)
from handlers.wallet import convert_str_date_to_ms

binpan_logger = LogManager(filename='./logs/binpan.log', name='binpan', info_level='INFO')

empty_agg_trades_msg = "Empty trades, please request using: get_agg_trades() method: Example: my_symbol.get_agg_trades()"
empty_atomic_trades_msg = "Empty atomic trades, please request using: get_atomic_trades() method: Example: my_symbol.get_atomic_trades()"


def _plotting():
    from handlers import plotting
    return plotting


class IndicatorsMixin:
    """Mixin that adds technical indicator methods to Symbol."""

    def ma(self,
           ma_name: str = 'ema',
           column_source: str = 'Close',
           inplace: bool = False,
           suffix: str = None,
           color: str or int = None,
           **kwargs):
        """
        Generic moving average method. Calls pandas_ta 'ma' method.

        `<https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/ma.py>`_

        :param str ma_name: A moving average supported by the generic pandas_ta "ma" function.
        :param str column_source: Name of column with data to be used.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :param kwargs: From https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/ma.py
        :return: pd.Series

        """
        binpan_logger.debug("This method is a sub-method used by sma() or ema(). PLease call it directly only if you know "
                            "what you are doing. It uses pandas_ta ma method.")
        if 'length' in kwargs.keys():
            if kwargs['length'] >= len(self.df):
                msg = f"BinPan Error: Ma window larger than data length."
                binpan_logger.error(msg)
                raise Exception(msg)
        if suffix:
            kwargs.update({'suffix': suffix})

        df = self.df.copy(deep=True)

        if ma_name == 'ema':
            from handlers.numba_tools import ema_numba
            ma_ = ema_numba(df[column_source].values, window=kwargs['length'])
            ma = pd.Series(data=ma_, index=df.index, name=f"EMA_{kwargs['length']}")

        elif ma_name == 'sma':
            from handlers.numba_tools import sma_numba
            ma_ = sma_numba(df[column_source].values, window=kwargs['length'])
            ma = pd.Series(data=ma_, index=df.index, name=f"SMA_{kwargs['length']}")

        else:
            raise BinPanException(f"BinPan Error: Moving average type '{ma_name}' not supported. Use 'ema' or 'sma'.")

        if inplace and self.is_new(ma):
            # plot ready
            column_name = str(ma.name)
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=str(column_name), row_position=1)  # overlaps are one

            self.df.loc[:, column_name] = ma

        return ma

    def sma(self, window: int = 21, column: str = 'Close', inplace=True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Generate technical indicator Simple Moving Average.

        :param int window: Rolling window including the current candles when calculating the indicator.
        :param str column: Column applied. Default is Close.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Color to show when plotting. It can be any color from plotly library or a number in the list of those.
            <https://community.plotly.com/t/plotly-colours-list/11730>
        :param kwargs: Optional plotly args from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/sma.py
        :return: pd.Series

        .. image:: images/indicators/sma.png
           :width: 1000
           :alt: Candles with some indicators

        """
        return self.ma(ma_name='sma', column_source=column, inplace=inplace, length=window, suffix=suffix, color=color, **kwargs)

    def ema(self, window: int = 21, column: str = 'Close', inplace=True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Generate technical indicator Exponential Moving Average.

        :param int window: Rolling window including the current candles when calculating the indicator.
        :param str column: Column applied. Default is Close.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Color to show when plotting. It can be any color from plotly library or index number in that list.
            <https://community.plotly.com/t/plotly-colours-list/11730>
        :param kwargs: Optional plotly args from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/ema.py
        :return: pd.Series

        .. image:: images/indicators/ema.png
           :width: 1000
           :alt: Candles with some indicators

        """
        if not color:
            color = 'skyblue'
        return self.ma(ma_name='ema', column_source=column, inplace=inplace, length=window, suffix=suffix, color=color, **kwargs)

    def supertrend(self, length: int = 10, multiplier: int = 3, inplace=True, suffix: str = None, colors: list = None, **kwargs):
        """
        Generate technical indicator Supertrend.

        :param int length: Rolling window including the current candles when calculating the indicator.
        :param int multiplier: Indicator multiplier applied.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: Defaults to red and green.
        :param kwargs: Optional plotly args from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/supertrend.py.
        :return: pd.DataFrame

        .. image:: images/indicators/supertrend.png
           :width: 1000
           :alt: Candles with some indicators

        """
        supertrend_df = supertrend_indicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'],
                                              length=length, multiplier=float(multiplier))
        supertrend_df.replace(0, np.nan, inplace=True)

        if inplace and self.is_new(supertrend_df):
            column_names = supertrend_df.columns
            self.row_counter += 1
            if not colors:
                colors = ['brown', 'blue', 'green', 'red']
            for i, col in enumerate(column_names):
                self.set_plot_color(indicator_column=col, color=colors[i])
                self.set_plot_color_fill(indicator_column=col, color_fill=False)
                if col.startswith("SUPERTs_"):
                    self.set_plot_row(indicator_column=col, row_position=1)  # overlaps are one
                elif col.startswith("SUPERTl_"):
                    self.set_plot_row(indicator_column=col, row_position=1)  # overlaps are one
                elif col.startswith("SUPERT_"):
                    self.set_plot_row(indicator_column=col, row_position=1)  # overlaps are one
                elif col.startswith("SUPERTd_"):
                    self.set_plot_row(indicator_column=col, row_position=self.row_counter)  # overlaps are one
            self.df = pd.concat([self.df, supertrend_df], axis=1)
        return supertrend_df

    def macd(self, fast: int = 12, slow: int = 26, smooth: int = 9, inplace: bool = True, suffix: str = '', colors: list = None, **kwargs):
        """
        Generate technical indicator Moving Average, Convergence/Divergence (MACD).

            https://www.investopedia.com/terms/m/macd.asp

        :param int fast: Fast rolling window including the current candles when calculating the indicator.
        :param int slow: Slow rolling window including the current candles when calculating the indicator.
        :param int smooth: Factor to apply a smooth in values. A smooth is a kind of moving average in short period like 3 or 9.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: A list of colors for the MACD dataframe columns. Is the color to show when plotting.
            It can be any color from plotly library or a number in the list of those. Default colors defined.

            <https://community.plotly.com/t/plotly-colours-list/11730>

        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/macd.py
        :return: pd.Series

        .. image:: images/indicators/macd.png
           :width: 1000
           :alt: Candles with some indicators

        """
        if not colors:
            colors = ['black', 'orange', 'green', 'blue']
        macd = macd_indicator(close=self.df['Close'], fast=fast, slow=slow, signal=smooth)
        zeros = macd.iloc[:, 0].copy()
        zeros.loc[:] = 0
        zeros.name = 'zeros'
        macd = pd.concat([zeros, macd], axis=1, ignore_index=False)

        if inplace and self.is_new(macd):
            self.row_counter += 1

            self.global_axis_group -= 1
            axis_identifier = f"y{self.global_axis_group}"  # for filling plots?

            for i, column_name in enumerate(macd.columns):
                col = macd[column_name]
                column_name = str(col.name) + suffix
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
                self.set_plot_axis_group(indicator_column=column_name, my_axis_group=axis_identifier)

                if column_name.startswith('MACDh_'):

                    splitted_dfs = df_splitter(data=macd, up_column=column_name, down_column='zeros')

                    self.set_plot_splitted_serie_couple(indicator_column_up=column_name, indicator_column_down='zeros',
                                                        splitted_dfs=splitted_dfs, color_up='rgba(35, 152, 33, 0.5)', color_down='rgba('
                                                                                                                                 '245, '
                                                                                                                                 '63, 39, '
                                                                                                                                 '0.5)')

                else:
                    self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
                    self.set_plot_filled_mode(indicator_column=column_name, fill_mode=None)

                self.df.loc[:, column_name] = col

        return macd

    def rsi(self, length: int = 14, inplace: bool = True, suffix: str = '', color: str or int = None):
        """
        Relative Strength Index (RSI).

            https://www.investopedia.com/terms/r/rsi.asp

        :param int length: Default is 21
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting from plotly list or index of color in that list.
        :return: A Pandas Series

        .. image:: images/indicators/rsi.png
           :width: 1000
           :alt: Candles with some indicators

        """

        # if self.is_numba:
        from handlers.numba_tools import rsi_numba
        rsi_ = rsi_numba(self.df['Close'].values, window=length)
        rsi = pd.Series(data=rsi_, index=self.df.index, name=f"RSI_{length}")
        # else:
        #     rsi = ta.rsi(close=self.df['Close'], length=length, **kwargs)

        column_name = str(rsi.name) + suffix

        if inplace and self.is_new(rsi):
            self.row_counter += 1
            if not color:
                color = 'orange'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = rsi
        return rsi

    def stoch_rsi(self, rsi_length: int = 14, k_smooth: int = 3, d_smooth: int = 3, inplace: bool = True, suffix: str = '',
                  colors: list = None, **kwargs) -> pd.DataFrame:
        """
        Stochastic Relative Strength Index (RSI) with a fast and slow exponential moving averages.

            https://www.investopedia.com/terms/s/stochrsi.asp

        :param int rsi_length: Default is 21
        :param int k_smooth: Smooth fast line with a moving average of some periods. default is 3.
        :param int d_smooth: Smooth slow line with a moving average of some periods. default is 3.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: Is the color to show when plotting.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/stochrsi.py
        :return: A Pandas DataFrame

        .. image:: images/indicators/stochrsi.png
           :width: 1000
           :alt: Candles with some indicators

        """
        if not colors:
            colors = ['orange', 'blue']
        stoch_df = stoch_rsi_indicator(close=self.df['Close'], rsi_length=rsi_length,
                                       stoch_length=rsi_length, k_smooth=k_smooth, d_smooth=d_smooth)
        if inplace and self.is_new(stoch_df):
            self.row_counter += 1
            for i, c in enumerate(stoch_df.columns):
                col = stoch_df[c]
                column_name = str(col.name) + suffix
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
                self.df.loc[:, column_name] = col
        return stoch_df

    def on_balance_volume(self, inplace: bool = True, suffix: str = '', color: str or int = None, **kwargs):
        """
        On balance indicator.

            https://www.investopedia.com/terms/o/onbalancevolume.asp

        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volume/obv.py
        :return: A Pandas Series

        .. image:: images/indicators/on_balance.png
           :width: 1000
           :alt: Candles with some indicators

        """

        on_balance = obv_indicator(close=self.df['Close'], volume=self.df['Volume'])

        column_name = str(on_balance.name) + suffix

        if inplace and self.is_new(on_balance):
            self.row_counter += 1

            if not color:
                color = 'pink'

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = on_balance

        return on_balance

    def accumulation_distribution(self, inplace: bool = True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Accumulation/Distribution indicator.

            https://www.investopedia.com/terms/a/accumulationdistribution.asp

        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volume/ad.py
        :return: A Pandas Series

        .. image:: images/indicators/ad.png
           :width: 1000
           :alt: Candles with some indicators

        """

        ad = ad_indicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], volume=self.df['Volume'])

        column_name = str(ad.name) + suffix

        if inplace and self.is_new(ad):
            self.row_counter += 1
            if not color:
                color = 'red'

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = ad

        return ad

    def vwap(self, anchor: str = "D", inplace: bool = True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Volume Weighted Average Price.

            https://www.investopedia.com/ask/answers/031115/why-volume-weighted-average-price-vwap-important-traders-and-analysts.asp

        :param str anchor: How to anchor VWAP. Depending on the index values, it will implement various Timeseries Offset Aliases
            as listed here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
            Default: "D", that means calendar day frequency.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/vwap.py
        :return: A Pandas Series

        .. image:: images/indicators/vwap.png
           :width: 1000
           :alt: Candles with some indicators
        """

        vwap = vwap_indicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], volume=self.df['Volume'])

        column_name = str(vwap.name) + suffix

        if inplace and self.is_new(vwap):
            # self.row_counter += 1
            if not color:
                color = 'darkgrey'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=1)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = vwap
        return vwap

    def atr(self, length: int = 14, inplace: bool = True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Average True Range.

            https://www.investopedia.com/terms/a/atr.asp

        :param str length: Window period to obtain ATR. Default is 14.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volatility/atr.py
        :return: A Pandas Series

        .. image:: images/indicators/atr.png
           :width: 1000
           :alt: Candles with some indicators

        """

        atr = atr_indicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], length=length)

        column_name = str(atr.name) + suffix

        if inplace and self.is_new(atr):
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = atr

        return atr

    def cci(self, length: int = 14, scaling: int = None, inplace: bool = True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Compute the Commodity Channel Index (CCI) for NIFTY based on the 14-day moving average.
        CCI can be used to determine overbought and oversold levels.

        - Readings above +100 can imply an overbought condition
        - Readings below −100 can imply an oversold condition.

        However, one should be careful because security can continue moving higher after the CCI indicator becomes
        overbought. Likewise, securities can continue moving lower after the indicator becomes oversold.

            https://blog.quantinsti.com/build-technical-indicators-in-python/

        :param str length: Window period to obtain ATR. Default is 14.
        :param str scaling: Scaling Constant. Default: 0.015.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/cci.py
        :return: A Pandas Series

        .. image:: images/indicators/cci.png
           :width: 1000
           :alt: Candles with some indicators

        """

        if scaling is None:
            scaling = 0.015
        cci = cci_indicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], length=length, c=scaling)

        column_name = str(cci.name) + suffix

        if inplace and self.is_new(cci):
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = cci

        return cci

    def eom(self, length: int = 14, divisor: int = 100000000, drift: int = 1, inplace: bool = True, suffix: str = '',
            color: str or int = None, **kwargs):
        """
        Ease of Movement (EMV) can be used to confirm a bullish or a bearish trend. A sustained positive Ease of Movement
        together with a rising market confirms a bullish trend, while a negative Ease of Movement values with falling
        prices confirms a bearish trend. Apart from using as a standalone indicator, Ease of Movement (EMV) is also used
        with other indicators in chart analysis.

            https://blog.quantinsti.com/build-technical-indicators-in-python/

        :param str length: The short period. Default: 14
        :param str divisor: Scaling Constant. Default is 100000000.
        :param str drift: The diff period. Default is 1
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volume/eom.py
        :return: A Pandas Series

        .. image:: images/indicators/eom.png
           :width: 1000
           :alt: Candles with some indicators

        """

        eom = eom_indicator(high=self.df['High'], low=self.df['Low'], volume=self.df['Volume'],
                            length=length, divisor=divisor)

        column_name = str(eom.name) + suffix

        if inplace and self.is_new(eom):
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = eom
        return eom

    def roc(self, length: int = 1, escalar: int = 100, inplace: bool = True, suffix: str = '', color: str or int = None, **kwargs):
        """
        The Rate of Change (ROC) is a technical indicator that measures the percentage change between the most recent price
        and the price "n" day's ago. The indicator fluctuates around the zero line.

                https://blog.quantinsti.com/build-technical-indicators-in-python/

        :param str length: The short period. Default: 1
        :param str escalar:  How much to magnify. Default: 100.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/roc.py
        :return: A Pandas Series

        .. image:: images/indicators/roc.png
           :width: 1000
           :alt: Candles with some indicators

        """

        roc = roc_indicator(close=self.df['Close'], length=length, scalar=escalar)

        column_name = str(roc.name) + suffix

        if inplace and self.is_new(roc):
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = roc
        return roc

    def bbands(self, length: int = 5, std: int = 2, ddof: int = 0, inplace: bool = True, suffix: str = '', colors: list = None,
               my_fill_color: str = 'rgba(47, 48, 56, 0.2)', **kwargs):
        """
        These bands consist of an upper Bollinger band and a lower Bollinger band and are placed two standard deviations
        above and below a moving average.
        Bollinger bands expand and contract based on the volatility. During a period of rising volatility, the bands widen,
        and they contract as the volatility decreases. Prices are considered to be relatively high when they move above
        the upper band and relatively low when they go below the lower band.

            https://blog.quantinsti.com/build-technical-indicators-in-python/

        :param int length: The short period. Default: 5
        :param int std: The long period. Default: 2
        :param int ddof: Degrees of Freedom to use. Default: 0
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: A list of colors for the indicator dataframe columns. Is the color to show when plotting.
            It can be any color from plotly library or a number in the list of those. Default colors defined.
            https://community.plotly.com/t/plotly-colours-list/11730
        :param str my_fill_color: An rgba color code to fill between bands area. https://rgbacolorpicker.com/
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volatility/bbands.py
        :return: pd.Series

        .. image:: images/indicators/bbands.png
           :width: 1000
           :alt: Candles with some indicators

        """
        if not colors:
            colors = ['red', 'orange', 'green']
        bbands = bbands_indicator(close=self.df['Close'], length=length, std=std, ddof=ddof)

        if inplace and self.is_new(bbands):
            self.global_axis_group -= 1
            axis_identifier = f"y{self.global_axis_group}"
            binpan_logger.debug(bbands.columns)
            for i, c in enumerate(bbands.columns):

                col = bbands[c]
                column_name = str(col.name)
                self.df.loc[:, column_name] = col
                if c.startswith('BBB') or c.startswith('BBP'):
                    continue
                self.set_plot_color(indicator_column=column_name, color=colors[i])

                if c.startswith('BBM'):
                    self.set_plot_color_fill(indicator_column=column_name, color_fill=my_fill_color)
                    self.set_plot_axis_group(indicator_column=column_name, my_axis_group=axis_identifier)
                    self.set_plot_filled_mode(indicator_column=column_name, fill_mode='tonexty')

                if c.startswith('BBU'):
                    self.set_plot_color_fill(indicator_column=column_name, color_fill=my_fill_color)
                    self.set_plot_axis_group(indicator_column=column_name, my_axis_group=axis_identifier)
                    self.set_plot_filled_mode(indicator_column=column_name, fill_mode='tonexty')

                self.set_plot_row(indicator_column=str(column_name), row_position=1)
        return bbands

    def stoch(self, k_length: int = 14, stoch_d=3, k_smooth: int = 1, inplace: bool = True, suffix: str = '', colors: list = None,
              **kwargs) -> pd.DataFrame:
        """
        Stochastic Oscillator with a fast and slow exponential moving averages.

            https://www.investopedia.com/terms/s/stochasticoscillator.asp

        :param int k_length: The Fast %K period. Default: 14
        :param int stoch_d: The Slow %K period. Default: 3
        :param int k_smooth: The Slow %D period. Default: 3
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: Is the color to show when plotting.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/stoch.py
        :return: A Pandas DataFrame

        .. image:: images/indicators/stoch_oscillator.png
           :width: 1000
           :alt: Candles with some indicators

        """
        if not colors:
            colors = ['orange', 'blue']
        stoch_df = stoch_indicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'],
                                   k=k_length, d=stoch_d, smooth_k=k_smooth)
        if inplace and self.is_new(stoch_df):
            self.row_counter += 1
            for i, c in enumerate(stoch_df.columns):
                col = stoch_df[c]
                column_name = str(col.name) + suffix
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
                self.df.loc[:, column_name] = col
        return stoch_df

    def ichimoku(self, tenkan: int = 9, kijun: int = 26, chikou_span: int = 26, senkou_cloud_base: int = 52, inplace: bool = True,
                 suffix: str = '', colors: list = None):
        """
        The Ichimoku Cloud is a collection of technical indicators that show support and resistance levels, as well as momentum and trend
        direction. It does this by taking multiple averages and plotting them on a chart. It also uses these figures to compute a "cloud"
        that attempts to forecast where the price may find support or resistance in the future.

            https://school.stockcharts.com/doku.php?id=technical_indicators:ichimoku_cloud

            https://www.youtube.com/watch?v=mCri-FFvZjo&list=PLv-cA-4O3y97HAd9OCvVKSfvQ8kkAGKlf&index=7

        :param int tenkan: The short period. It's the half sum of max and min price in the window. Default: 9
        :param int kijun: The long period. It's the half sum of max and min price in the window. Default: 26
        :param int chikou_span: Close of the next 26 bars. Util for spotting what happened with other ichimoku lines and what happened
           before Default: 26.
        :param senkou_cloud_base: Period to obtain kumo cloud base line. Default is 52.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: A list of colors for the indicator dataframe columns. Is the color to show when plotting.
            It can be any color from plotly library or a number in the list of those. Default colors defined.
            https://community.plotly.com/t/plotly-colours-list/11730

        :return: pd.Series

        .. image:: images/indicators/ichimoku.png
           :width: 1000
           :alt: Candles with some indicators

        """
        if not colors:
            colors = ['orange', 'skyblue', 'grey', 'green', 'red']
        ichimoku_data = ichimoku(data=self.df, tenkan=tenkan, kijun=kijun, chikou_span=chikou_span, senkou_cloud_base=senkou_cloud_base,
                                 suffix=suffix)

        if inplace and self.is_new(ichimoku_data):
            self.global_axis_group -= 1
            axis_identifier = f"y{self.global_axis_group}"

            # expand index
            missing_index = set(ichimoku_data.index) - set(self.df.index)
            self.df = self.df.reindex(self.df.index.union(missing_index))

            binpan_logger.debug(ichimoku_data.columns)

            for i, column_name in enumerate(ichimoku_data.columns):
                column_name = str(column_name) + suffix

                col_data = ichimoku_data[column_name]
                self.df.loc[:, column_name] = col_data

                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=str(column_name), row_position=1)

                if column_name.startswith(f'Ichimoku_cloud_{senkou_cloud_base}'):
                    self.set_plot_axis_group(indicator_column=column_name, my_axis_group=axis_identifier)

                    other_cloud_columns = [c for c in ichimoku_data.columns if c.startswith('Ichimoku_cloud_')]
                    col_idx = other_cloud_columns.index(column_name) - 1
                    pre_col_name = other_cloud_columns[col_idx]

                    splitted_dfs = df_splitter(data=ichimoku_data, up_column=pre_col_name, down_column=column_name)

                    self.set_plot_splitted_serie_couple(indicator_column_up=pre_col_name, indicator_column_down=column_name,
                                                        splitted_dfs=splitted_dfs, color_up='rgba(35, 152, 33, 0.5)', color_down='rgba('
                                                                                                                                 '245, '
                                                                                                                                 '63, 39, '
                                                                                                                                 '0.5)')

        return ichimoku_data

    def alternating_fractal(self, max_period: int = None, inplace: bool = True, overlap_plot=True, with_trend: bool = True,
                            suffix: str = '', colors: list = None) -> tuple[pd.Series | None, float | None, float | None]:
        """
        Obtains the minim value for fractal_w periods as fractal is pure alternating from max to min to max etc. In other words,
        max and mins alternates in regular rhythm without any tow max or two mins consecutively.

        This custom indicator shows the minimum period in finding a pure alternating fractal. It is some kind of rhythm in price
        indicator, the most period needed, the slow price rhythm.

        :param int max_period: Default is len of dataframe. This method will check from 2 to the max period value to find a alternating
         max to mins.
        :param bool inplace: Make it permanent in the instance or not.
        :param bool overlap_plot: If True, it will overlap the indicator plot with the price plot.
        :param bool with_trend: If true, it will return maximums diff mean and minimums diff mean also.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: A list of colors for the indicator dataframe columns. Is the color to show when plotting. It can be any color
          from plotly library or a number in the list of those. Default colors defined.
          https://community.plotly.com/t/plotly-colours-list/11730
        :return pd.DataFrame: A dataframe with two columns, one with 1 or -1 for local max or local min to tag,
         and other with price values for that points. Alternatively, maximums and minimums diff mean will be returned.

        """
        fractal = alternating_fractal_indicator(self.df, suffix=suffix, max_period=max_period)
        max_mean, min_mean = None, None
        if type(fractal) != pd.DataFrame:
            binpan_logger.warning(f'No pure alternating fractal found for {max_period} periods')
            return None, None, None
        if with_trend:
            max_mean, min_mean = fractal_trend_indicator(df=self.df, period=max_period, fractal=fractal, suffix=suffix)
            if max_mean > 0 and min_mean > 0:
                msg = f"Increasing maxima and increasing minima. uptrend"
            elif max_mean < 0 and min_mean < 0:
                msg = f"Decreasing maxima and decreasing minima. downtrend"
            elif max_mean < 0 < min_mean:
                msg = f"Decreasing maxima and increasing minima, not clear trend"
            elif max_mean > 0 > min_mean:
                msg = f"Increasing maxima and decreasing minima, not clear trend"
            else:
                msg = f"No trend detected"
            binpan_logger.info(msg)
            print(msg)

            binpan_logger.info(f"Max mean: {max_mean} Min mean: {min_mean}")
        if not colors:
            colors = ['black', 'black']

        if inplace and self.is_new(fractal):
            binpan_logger.debug(fractal.columns)

            for i, column_name in enumerate(fractal.columns):
                if overlap_plot and i > 0:
                    col_data = fractal[column_name].ffill()
                else:
                    self.row_counter += 1
                    col_data = fractal[column_name]
                self.df.loc[:, column_name] = col_data
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                # self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)
                if overlap_plot and i > 0:
                    self.set_plot_row(indicator_column=str(column_name), row_position=1)  # overlaps are one
                else:
                    self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)

        return fractal, max_mean, min_mean

    def fractal(self, period: int = 5, inplace: bool = True, overlap_plot=True, suffix: str = '', colors: list = None):
        """
        The fractal indicator is based on a simple price pattern that is frequently seen in financial markets. Outside of trading, a fractal
        is a recurring geometric pattern that is repeated on all time frames. From this concept, the fractal indicator was devised.
        The indicator isolates potential turning points on a price chart. It then draws arrows to indicate the existence of a pattern.

        https://www.investopedia.com/terms/f/fractal.asp

        From: https://codereview.stackexchange.com/questions/259703/william-fractal-technical-indicator-implementation

        :param int period: Default is 2. Count of neighbour candles to match max or min tags.
        :param bool inplace: Make it permanent in the instance or not.
        :param bool overlap_plot: If True, it will overlap the indicator plot with the price plot.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: A list of colors for the indicator dataframe columns. Is the color to show when plotting.
            It can be any color from plotly library or a number in the list of those. Default colors defined.
            https://community.plotly.com/t/plotly-colours-list/11730

        :return pd.Series: A serie with 1 or -1 for local max or local min to tag.

        """
        if not colors:
            colors = ['black', 'black']
        fractal = fractal_w_indicator(df=self.df, period=period, suffix=suffix, fill_with_zero=True)

        if inplace and self.is_new(fractal):

            binpan_logger.debug(fractal.columns)

            for i, column_name in enumerate(fractal.columns):

                if overlap_plot and i > 0:
                    col_data = fractal[column_name].ffill()
                else:
                    self.row_counter += 1
                    col_data = fractal[column_name]

                self.df.loc[:, column_name] = col_data

                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                if overlap_plot and i > 0:
                    self.set_plot_row(indicator_column=str(column_name), row_position=1)  # overlaps are one
                else:
                    self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)

        return fractal

    def get_market_profile(self, bins: int = 100, hours: int = None, minutes: int = None, startTime: int or str = None,
                           endTime: int or str = None, from_agg_trades=False, from_atomic_trades=False,
                           time_zone: str = None) -> pd.DataFrame or None:
        """
        Generates a market profile dataframe from trade or kline data. The market profile is a histogram of trading
        volumes at different price levels.

        :param bins: The number of price levels (bins) to include in the market profile.
        :param hours: If specified, only the last 'hours' hours of data are used to generate the market profile.
        :param minutes: If specified, only the last 'minutes' minutes of data are used to generate the market profile.
        :param startTime: If specified, only data after this timestamp or date (in format %Y-%m-%d %H:%M:%S) are used.
        :param endTime: If specified, only data before this timestamp or date (in format %Y-%m-%d %H:%M:%S) are used.
        :param from_agg_trades: If True, aggregated trades data are used to generate the market profile.
        :param from_atomic_trades: If True, atomic trades data are used to generate the market profile.
        :param time_zone: The time zone to use for time index conversion (e.g., "Europe/Madrid").

        :return: A DataFrame representing the market profile, or None if no suitable data are available.
        """
        try:
            assert not (from_agg_trades and from_atomic_trades)
        except AssertionError:
            raise BinPanException(f"Please specify just one source of data, atomic trades or aggregated, not both.")

        if not self.market_profile_df.empty:
            binpan_logger.info(f"Market profile already generated. Updating data: startTime={startTime}, endTime={endTime}, "
                               f"hours={hours}, minutes={minutes}, bins={bins}, time_zone={time_zone}")
            del self.market_profile_df

        if time_zone:
            self.time_zone = time_zone
        if startTime:
            convert_str_date_to_ms(date=startTime, time_zone=self.time_zone)
        if endTime:
            convert_str_date_to_ms(date=endTime, time_zone=self.time_zone)
        if hours:
            startTime = int(time() * 1000) - (1000 * 60 * 60 * hours)
        elif minutes:
            startTime = int(time() * 1000) - (1000 * 60 * minutes)

        if from_agg_trades:
            if self.agg_trades.empty:
                binpan_logger.info(empty_agg_trades_msg)
                return
        if from_atomic_trades:
            if self.atomic_trades.empty:
                binpan_logger.info(empty_atomic_trades_msg)
                return

        if from_agg_trades:
            _df = self.agg_trades.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            self.market_profile_df = market_profile_from_trades_grouped(df=_df, num_bins=bins)
        elif from_atomic_trades:
            _df = self.atomic_trades.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            self.market_profile_df = market_profile_from_trades_grouped(df=_df, num_bins=bins)
        else:
            _df = self.df.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            binpan_logger.info(f"Using klines data. For deeper info add trades data, example: my_symbol.get_agg_trades()")
            self.market_profile_df = market_profile_from_klines_grouped(df=_df, num_bins=bins)
        return self.market_profile_df

    def get_maker_taker_buy_ratios(self, window: int = 14, inplace=True, colors: list = None, suffix: str = "",
                                   nans_to_zeros=True) -> pd.DataFrame:
        """
        Generates the makers versus makers+takers volume ratio by each_kline. Also adds a moving average of the ratio.

        :param int window: The window of the moving average.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int colors: A colors list. Default is ["orange", "skyblue"].
        :param bool nans_to_zeros: If True, NaNs are converted to zeros.
        :return: A pandas series with the ratio and the moving average.
        """

        df = self.df.copy(deep=True)
        df = df.sort_index(ascending=True)

        ratios = df['Taker buy base volume'] / df["Volume"]

        if nans_to_zeros:
            binpan_logger.info(f"Maker vs Taker buy ratio: {ratios.isna().sum()} NaNs found. Converting to zeros.")
            ratios.fillna(0, inplace=True)
        else:
            binpan_logger.info(f"Maker vs Taker buy ratio: {ratios.isna().sum()} NaNs found. Use nans_to_zeros=True to convert to zeros.")

        ratios.name = "Ratio_Taker/Maker_buy" + suffix

        ema = ratios.ewm(span=window, adjust=False, min_periods=window).mean()
        ema.name = f"Ratio_Taker/Maker_buy_EMA_{window}" + suffix

        if inplace and self.is_new(ratios):
            if not colors:
                colors = ["orange", "skyblue"]
            self.row_counter += 1

            for i, new_col in enumerate([ratios, ema]):
                column_name = str(new_col.name)
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
                self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)
                self.df.loc[:, column_name] = new_col

        return pd.DataFrame({ratios.name: ratios, ema.name: ema}, index=self.df.index).sort_index(ascending=True)

    def get_taker_maker_ratio_profile(self, bins: int = 100, hours: int = None, minutes: int = None, startTime: int or str = None,
                                      endTime: int or str = None, from_agg_trades=False, from_atomic_trades=False,
                                      time_zone: str = None) -> pd.DataFrame or None:
        """
        Generates a market profile of the makers versus makers+takers volume ratio by each_kline.

        :param bins: The number of price levels (bins) to include in the market profile.
        :param hours: If specified, only the last 'hours' hours of data are used to generate the market profile.
        :param minutes: If specified, only the last 'minutes' minutes of data are used to generate the market profile.
        :param startTime: If specified, only data after this timestamp or date (in format %Y-%m-%d %H:%M:%S) are used.
        :param endTime: If specified, only data before this timestamp or date (in format %Y-%m-%d %H:%M:%S) are used.
        :param from_agg_trades: If True, aggregated trades data are used to generate the market profile.
        :param from_atomic_trades: If True, atomic trades data are used to generate the market profile.
        :param time_zone: The time zone to use for time index conversion (e.g., "Europe/Madrid").
        :return: A DataFrame representing the market profile, or None if no suitable data are available.
        """
        if self.market_profile_df.empty:
            self.market_profile_df = self.get_market_profile(bins=bins, hours=hours, minutes=minutes, startTime=startTime,
                                                             endTime=endTime, from_agg_trades=from_agg_trades,
                                                             from_atomic_trades=from_atomic_trades, time_zone=time_zone)
        return self.market_profile_df['Taker buy base volume'] / (
                self.market_profile_df['Taker buy base volume'] + self.market_profile_df['Maker buy base volume'])

    @staticmethod
    def pandas_ta_indicator(name: str, **kwargs):
        """
                Calls any indicator in pandas_ta library with function name as first argument and any kwargs the function will use.

                Generic calls are not added to object, just returned.

                More info: https://github.com/twopirllc/pandas-ta

                :param str name: A function name. In example: 'massi' for Mass Index or 'rsi' for RSI indicator.
                :param kwargs: Arguments for the requested indicator. Review pandas_ta info: https://github.com/twopirllc/pandas-ta#features
                :return: Whatever returns pandas_ta

                Example:

                .. code-block::

                  sym = binpan.Symbol(symbol='LUNCBUSD', tick_interval='1m')

                  sym.pandas_ta_indicator(name='ichimoku', **{
                                                            'high': sym.df['High'],
                                                            'low': sym.df['Low'],
                                                            'close': sym.df['Close'],
                                                            'tenkan': 9,
                                                            'kijun ': 26,
                                                            'senkou ': 52})


                        (                              ISA_9    ISB_26     ITS_9    IKS_26    ICS_26
                         LUNCBUSD 1m UTC
                         2022-10-06 23:27:00+00:00       NaN       NaN       NaN       NaN  0.000285
                         2022-10-06 23:28:00+00:00       NaN       NaN       NaN       NaN  0.000285
                         2022-10-06 23:29:00+00:00       NaN       NaN       NaN       NaN  0.000285
                         2022-10-06 23:30:00+00:00       NaN       NaN       NaN       NaN  0.000285
                         2022-10-06 23:31:00+00:00       NaN       NaN       NaN       NaN  0.000285
                         ...                             ...       ...       ...       ...       ...
                         2022-10-07 16:01:00+00:00  0.000292  0.000293  0.000291  0.000291       NaN
                         2022-10-07 16:02:00+00:00  0.000292  0.000293  0.000292  0.000291       NaN
                         2022-10-07 16:03:00+00:00  0.000292  0.000293  0.000292  0.000291       NaN
                         2022-10-07 16:04:00+00:00  0.000292  0.000293  0.000292  0.000291       NaN
                         2022-10-07 16:05:00+00:00  0.000292  0.000293  0.000292  0.000291       NaN

                         [999 rows x 5 columns],
                                                       ISA_9    ISB_26
                         2022-10-10 16:05:00+00:00  0.000292  0.000293
                         2022-10-11 16:05:00+00:00  0.000292  0.000293
                         2022-10-12 16:05:00+00:00  0.000292  0.000293
                         2022-10-13 16:05:00+00:00  0.000292  0.000293
                         2022-10-14 16:05:00+00:00  0.000292  0.000293
                         ...                             ...       ...
                         2022-11-08 16:05:00+00:00  0.000291  0.000292
                         2022-11-09 16:05:00+00:00  0.000291  0.000292
                         2022-11-10 16:05:00+00:00  0.000292  0.000292
                         2022-11-11 16:05:00+00:00  0.000292  0.000292
                         2022-11-14 16:05:00+00:00  0.000292  0.000292

                         [26 rows x 2 columns])

                """
        import warnings
        warnings.warn(
            "pandas_ta_indicator() está deprecado. pandas_ta ha sido eliminado. "
            "Usa los métodos nativos de Symbol (ema, sma, rsi, macd, supertrend, etc.).",
            DeprecationWarning, stacklevel=2)
        raise BinPanException(
            f"Indicator '{name}' not available. pandas_ta has been removed. "
            f"Use the native Symbol methods instead (ema, sma, rsi, macd, supertrend, etc.).")

    def support_resistance(self,
                           from_atomic: bool = False,
                           from_aggregated: bool = False,
                           max_clusters: int = 5,
                           by_quantity: float = None,
                           simple: bool = True,
                           inplace=True,
                           colors: list = None) -> tuple[list[float], list[float]]:
        """
        Calculate support and resistance levels for the Symbol based on either atomic trades or aggregated trades.

        :param bool from_atomic: If True, support and resistance levels will be calculated using atomic trades.
        :param bool from_aggregated: If True, support and resistance levels will be calculated using aggregated trades.
        :param int max_clusters: If passed, fixes count of levels of support and resistance. Default is 5.
        :param float by_quantity: It takes each price into account by how many times the specified quantity appears in "Quantity" column.
        :param bool simple: If True, it will calculate support and resistance levels merged. Just levels. Default is True.
        :param bool inplace: If True, it will replace the current dataframe with the new one. Default is True.
        :param list colors: A list of colors for the indicator dataframe columns. Is the color to show when plotting.
         It can be any color from plotly library or a number in the list of those. Default colors defined.
         https://community.plotly.com/t/plotly-colours-list/11730
        :return: A tuple containing two lists: the first list contains support levels, and the second list contains resistance levels.
        """

        if from_atomic:
            from_data = "atomic trades"
            if self.atomic_trades.empty:
                print(f"Please add atomic trades first: my_symbol.get_atomic_trades()")
            else:
                if not by_quantity:
                    by_quantity = np.mean(self.atomic_trades['Quantity'].values)
                if simple:
                    self.support_lines = support_resistance_levels_merged(self.atomic_trades,
                                                                          max_clusters=max_clusters,
                                                                          by_quantity=by_quantity,
                                                                          by_klines=False)
                    self.resistance_lines = []
                else:
                    self.support_lines, self.resistance_lines = support_resistance_levels(self.atomic_trades,
                                                                                          max_clusters=max_clusters,
                                                                                          by_quantity=by_quantity,
                                                                                          by_klines=False)
        elif from_aggregated:
            from_data = "aggregated trades"
            if self.agg_trades.empty:
                print(f"Please add aggregated trades first: my_symbol.get_agg_trades()")
            else:
                if not by_quantity:
                    by_quantity = np.mean(self.agg_trades['Quantity'].values)

                if simple:
                    self.support_lines = support_resistance_levels_merged(self.agg_trades,
                                                                          max_clusters=max_clusters,
                                                                          by_quantity=by_quantity,
                                                                          by_klines=False)
                    self.resistance_lines = []
                else:
                    self.support_lines, self.resistance_lines = support_resistance_levels(self.agg_trades,
                                                                                          max_clusters=max_clusters,
                                                                                          by_quantity=by_quantity,
                                                                                          by_klines=False)
        else:  # with klines
            from_data = "klines"
            if not by_quantity:
                by_quantity = np.mean(self.df['Trades'].values)
            if simple:
                self.support_lines = support_resistance_levels_merged(self.df,
                                                                      max_clusters=max_clusters,
                                                                      by_quantity=by_quantity,
                                                                      by_klines=True)
                self.resistance_lines = []
            else:
                self.support_lines, self.resistance_lines = support_resistance_levels(self.df,
                                                                                      max_clusters=max_clusters,
                                                                                      by_quantity=by_quantity,
                                                                                      by_klines=True)

        if inplace:
            # update data
            binpan_logger.info(f"Updating support_resistance levels for {self.symbol} from {from_data}")
            if [c for c in self.df.columns if "Support" in c]:
                self.delete_indicator_family(indicator_name_root="Support")
            if [c for c in self.df.columns if "Resistance" in c]:
                self.delete_indicator_family(indicator_name_root="Resistance")
            if simple:
                sup_cols = [f"Support_Resistance_{k + 1}" for k in range(len(self.support_lines))]
                res_cols = []
            else:
                sup_cols = [f"Support_{k + 1}" for k in range(len(self.support_lines))]
                res_cols = [f"Resistance_{k + 1}" for k in range(len(self.resistance_lines))]

            for i, c in enumerate(sup_cols):
                self.df.loc[:, c] = self.support_lines[i]
            for i, c in enumerate(res_cols):
                self.df.loc[:, c] = self.resistance_lines[i]

            # add plot info
            if not colors:
                colors = ["blue" for _ in sup_cols] + ["red" for _ in res_cols]

            for i, column_name in enumerate(sup_cols + res_cols):
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=str(column_name), row_position=1)  # overlaps are one  # return self.df
        return self.support_lines, self.resistance_lines

    def rolling_support_resistance(self,
                                   minutes_window: int = None,
                                   time_steps_minutes: int = None,
                                   discrete_interval: str = None,
                                   from_atomic: bool = False,
                                   from_aggregated: bool = False,
                                   max_clusters: int = 5,
                                   by_quantity: bool = True,
                                   simple: bool = True,
                                   inplace: bool = True,
                                   delayed: int = 0,
                                   colors: list = None) -> pd.DataFrame or None:
        """
        Calculate support and resistance levels for the Symbol based on either atomic trades or aggregated trades in a rolling window. Also
        from klines supported, but less accurate. It returns a pandas dataframe with each column representing ordered levels from lower to
        higher for support and resistance. The function iterates in steps of a minutes quantity or a discrete interval.

        If discrete_interval is passed, it will ignore time_steps_minutes and minutes_window and will use this interval to calculate the
        rolling support and resistance minutes_window and time_steps_minutes. It can be any of the binance kline ones: '1m', '3m', '5m',
        '15m',
        '30m', '1h', '2h', '4h', etc

        The parameter delayed is useful when you want to calculate the rolling support and resistance with a delay. For example, if you
        want to calculate the rolling support and resistance with the last 5 minutes of data, but you want to project it 5 minutes
        after the last minute of the window, you can pass delayed=1 (1 step of the interval selected). Useful for projecting support and
        resistance levels in the future.

        If simple parameter is True, it will calculate support and resistance levels merged. Just levels. Default is True.

        Example: If you want to calculate the rolling support and resistance with and interval of 24h and a delayed of 1, this will add
        past 24h
        support and resistance levels to the current dataframe

            .. code-block::

                my_symbol.rolling_support_resistance(discrete_interval='1d', delayed=1)

        .. image:: images/rolling_support_resistance.png

        Example: Not discrete mode but same 24 h intervals and not delayed.

            .. code-block::

                sym.rolling_support_resistance(minutes_window=24*60, time_steps_minutes=24*60, max_clusters=5)

        .. image:: images/rolling_support_resistance_2.png


        :param int minutes_window: A rolling window of time in minutes. Whe using trades, it will calculate window by time index.
        :param int time_steps_minutes: Loop steps in minutes. Default is 10. Each step will calculate a new window data.
        :param str discrete_interval: If passed, it will ignore time_steps_minutes and minutes_window and will use this interval to
            calculate the rolling support and resistance. It can be any of the following: '1m', '3m', '5m', '15m', '30m', '1h', '2h',
            '4h', etc
        :param bool from_atomic: If True, support and resistance levels will be calculated using atomic trades.
        :param bool from_aggregated: If True, support and resistance levels will be calculated using aggregated trades.
        :param int max_clusters: If passed, fixes count of levels of support and resistance. Default is 5.
        :param float by_quantity: It takes each price into account by how many times the specified quantity appears in "Quantity" column.
        :param bool simple: If True, it will calculate support and resistance levels merged. Just levels. Default is True.
        :param bool inplace: If True, it will replace the current dataframe with the new one. Default is True.
        :param int delayed: If passed, it will project the rolling support and resistance levels in the future. Default is 0  and means 0
        windows projected in the future.
        :param list colors: A list of colors for the indicator dataframe columns. Is the color to show when plotting.
         It can be any color from plotly library or a number in the list of those. Default colors defined.
         https://community.plotly.com/t/plotly-colours-list/11730
        :return pd.DataFrame: A pandas dataframe with each column representing ordered levels from higher to lower for support and
         resistance.
        """
        binpan_logger.info(f"Each {time_steps_minutes} minutes, support and resistance will be calculated with the last {minutes_window} "
                           f"minutes  data. Or by a discrete interval of {discrete_interval}.")
        by_klines = False

        if from_atomic:
            # generar columnas de minutos pasados
            df = self.atomic_trades.copy(deep=True)
        elif from_aggregated:
            df = self.agg_trades.copy(deep=True)
        else:
            df = self.df.copy(deep=True)
            by_klines = True

        try:
            assert isinstance(df.index, pd.DatetimeIndex), "Index is not DatetimeIndex"
        except AssertionError as e:
            binpan_logger.error(f"BinPan rolling_support_resistance error: {e}")
            return None

        result = pd.DataFrame(index=df.index)
        result.index.name = f"Rolling support resistance {df.index.name}"
        sup_cols = [f"Support_{k + 1}" for k in range(max_clusters)]
        res_cols = [f"Resistance_{k + 1}" for k in range(max_clusters)]

        if discrete_interval:
            pandas_interval = pandas_freq_tick_interval[discrete_interval]
            discrete_index = df.resample(pandas_interval).first().index

            # Extender el último índice para incluir el intervalo futuro. Util para delayed
            last_index = discrete_index[-1] + pd.to_timedelta(pandas_interval)
            extended_discrete_index = discrete_index.union([last_index])

            update_time_ranges = [
                (extended_discrete_index[i], extended_discrete_index[i + 1] - pd.Timedelta(seconds=1))
                for i in range(len(extended_discrete_index) - 1)
            ]

        else:
            # Lógica para calcular rangos de tiempo deslizantes
            delta_step = f"{time_steps_minutes}T"
            update_time_ranges = get_dataframe_time_index_ranges(data=df, interval=delta_step)
            update_time_ranges = remove_initial_included_ranges(time_ranges=update_time_ranges, initial_minutes=minutes_window)

        # loop por cada ventana de tiempo
        for i in range(len(update_time_ranges)):
            previous_start, previous_end = update_time_ranges[i - delayed]
            current_start, current_end = update_time_ranges[i]
            df_window = df.loc[previous_start:previous_end]

            # Aplica la función hipotética al fragmento del DataFrame
            if simple:
                s_lines = support_resistance_levels_merged(df=df_window,
                                                           max_clusters=max_clusters,
                                                           by_quantity=by_quantity,
                                                           by_klines=by_klines)
                r_lines = []
            else:
                s_lines, r_lines = support_resistance_levels(df=df_window,
                                                             max_clusters=max_clusters,
                                                             by_quantity=by_quantity,
                                                             by_klines=by_klines)
            for k, sup in enumerate(s_lines):
                result.loc[current_start:current_end, sup_cols[k]] = sup
            for k, res in enumerate(r_lines):
                result.loc[current_start:current_end, res_cols[k]] = res

        if inplace:
            # update data
            binpan_logger.info(f"Updating rolling_support_resistance_df: {sup_cols}, {res_cols} in simple mode = {simple}")
            del self.rolling_support_resistance_df
            self.rolling_support_resistance_df = result  # before reindexing for klines integration. remember this can be from trades

            # update data
            self.delete_indicator_family(indicator_name_root="Support")
            self.delete_indicator_family(indicator_name_root="Resistance")

            # eliminación de duplicados cunado proviene de trades
            if from_atomic or from_aggregated:
                result = result.loc[~result.index.duplicated(keep='first')]
                result = result.reindex(self.df.index, method='ffill')
            self.df = self.df.merge(result, left_index=True, right_index=True, how='left')

            if not colors:
                colors = ["blue" for _ in sup_cols] + ["red" for _ in res_cols]

            for i, column_name in enumerate(result.columns):
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=str(column_name), row_position=1)  # overlaps are one

        return self.rolling_support_resistance_df

    def time_centroids(self,
                       from_atomic: bool = False,
                       from_aggregated: bool = False,
                       max_clusters: int = 5,
                       by_quantity: float = None,
                       simple: bool = True) -> tuple[list[float], list[float]]:
        """
        Calculate centroids for timestamps of more activity in taker buys or takers sells.

        :param bool from_atomic: If True, centroids will be calculated using atomic trades.
        :param bool from_aggregated: If True, centroids will be calculated using aggregated trades.
        :param int max_clusters: If passed, fixes count of levels of support and resistance. Default is 5.
        :param float by_quantity: It takes each price into account by how many times the specified quantity appears in "Quantity" column.
        :param bool simple: If True, it will calculate centroids merged. Just levels, not buy or sell centroids. Default is True.
        :return: A tuple containing two lists: the first list contains buy centroids, and the second list contains sell centroids.

        .. image:: images/indicators/time_action.png
           :width: 1000

        """

        if from_atomic:
            from_data = "atomic trades"
            if self.atomic_trades.empty:
                print(f"Please add atomic trades first: my_symbol.get_atomic_trades()")
            else:
                if not by_quantity:
                    by_quantity = np.mean(self.atomic_trades['Quantity'].values)
                if simple:
                    self.blue_timestamps, self.red_timestamps = time_active_zones(self.atomic_trades, max_clusters=max_clusters,
                                                                                  by_quantity=by_quantity,
                                                                                  simple=simple)

                else:
                    self.blue_timestamps, self.red_timestamps = time_active_zones(self.atomic_trades, max_clusters=max_clusters,
                                                                                  by_quantity=by_quantity, simple=simple)
        elif from_aggregated:
            from_data = "aggregated trades"
            if self.agg_trades.empty:
                print(f"Please add aggregated trades first: my_symbol.get_agg_trades()")
            else:
                if not by_quantity:
                    by_quantity = np.mean(self.agg_trades['Quantity'].values)

                if simple:
                    self.blue_timestamps, self.red_timestamps = time_active_zones(self.agg_trades,
                                                                                  max_clusters=max_clusters,
                                                                                  by_quantity=by_quantity,
                                                                                  simple=simple)

                else:
                    self.blue_timestamps, self.red_timestamps = time_active_zones(self.agg_trades,
                                                                                  max_clusters=max_clusters,
                                                                                  by_quantity=by_quantity, simple=simple)
        else:  # with klines
            from_data = "klines"
            if not by_quantity:
                by_quantity = np.mean(self.df['Trades'].values)
            if simple:
                self.blue_timestamps, self.red_timestamps = time_active_zones(self.df,
                                                                              max_clusters=max_clusters,
                                                                              by_quantity=by_quantity,
                                                                              simple=simple)

            else:
                self.blue_timestamps, self.red_timestamps = time_active_zones(self.df,
                                                                              max_clusters=max_clusters,
                                                                              by_quantity=by_quantity, simple=simple)
        binpan_logger.info(f"Time centroids added from {from_data}")
        return self.blue_timestamps, self.red_timestamps
