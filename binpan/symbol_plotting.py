"""
Plotting methods for Symbol class.
"""
from __future__ import annotations

from random import choice
from time import time

import pandas as pd

from .core.exceptions import BinPanException
from .core.logs import LogManager
from .core.time_helper import tick_seconds
from .api.wallet_api import convert_str_date_to_ms
from .analysis.indicators import zoom_cloud_indicators, market_profile_from_klines_melt

binpan_logger = LogManager(filename='./logs/binpan.log', name='binpan', info_level='INFO')

empty_agg_trades_msg = "Empty trades, please request using: get_agg_trades() method: Example: my_symbol.get_agg_trades()"
empty_atomic_trades_msg = "Empty atomic trades, please request using: get_atomic_trades() method: Example: my_symbol.get_atomic_trades()"


def _plotting():
    from binpan.plotting import charts
    return charts


class SymbolPlotting:
    """Plotting methods for Symbol."""

    def set_plot_row(self, indicator_column: str = None, row_position: int = None):
        """
        Internal control formatting plots. Can be used to change plot subplot row of an indicator.

        :param str indicator_column: column name
        :param row_position: reassign row_position to column name
        :return dict: columns with its assigned row_position in subplots when charts.

        """
        if indicator_column and row_position:
            self.row_control.update({indicator_column: row_position})
        return self.row_control

    def set_plot_color(self, indicator_column: str = None, color: int | str = None) -> dict:
        """
        Internal control formatting plots. Can be used to change plot color of an indicator.

        :param str indicator_column: column name
        :param color: reassign color to column name
        :return dict: columns with its assigned colors when charts.

        """
        if indicator_column and color:
            if type(color) == int:
                self.color_control.update({indicator_column: color})
            elif color in _plotting().plotly_colors:
                self.color_control.update({indicator_column: color})
            else:
                self.color_control.update({indicator_column: choice(_plotting().plotly_colors)})
        elif indicator_column:
            self.color_control.update({indicator_column: choice(_plotting().plotly_colors)})
        return self.color_control

    def set_plot_color_fill(self, indicator_column: str = None, color_fill: str | bool = None) -> dict:
        """
        Internal control formatting plots. Can be used to change plot color of an indicator.

        :param str indicator_column: column name
        :param color_fill: Color can be forced to fill to zero line. For transparent colors use rgba string code to define color.
         Example for transparent green 'rgba(26,150,65,0.5)' or transparent red 'rgba(204,0,0,0.5)'
        :return dict: columns with its assigned colors when charts.

        """
        if indicator_column and color_fill:
            if type(color_fill) == int:
                self.color_fill_control.update({indicator_column: _plotting().plotly_colors[color_fill]})
            elif color_fill in _plotting().plotly_colors or color_fill.startswith('rgba'):
                self.color_fill_control.update({indicator_column: color_fill})
            else:
                self.color_fill_control.update({indicator_column: None})
        elif indicator_column:
            self.color_fill_control.update({indicator_column: None})
        return self.color_fill_control

    def set_plot_filled_mode(self, indicator_column: str = None, fill_mode: str = None) -> dict:
        """
        Internal control formatting plots. Can be used to change plot filling mode for pairs of indicators when.

        :param str indicator_column: column name
        :param fill_mode: Fill mode for indicator. Color can be forced to fill to zero line with "tozeroy" or between two indicators in same
           axis group with "tonexty".
        :return dict: columns with its assigned fill mode.

        """
        if fill_mode:
            try:
                assert fill_mode == 'tonexty' or fill_mode == 'tozeroy'
            except Exception:
                print(f"Fill mode need to be 'tonexty' or 'tozeroy'")
                return self.indicators_filled_mode
        if indicator_column and fill_mode:
            self.indicators_filled_mode.update({indicator_column: fill_mode})
        return self.indicators_filled_mode

    def set_plot_axis_group(self, indicator_column: str = None, my_axis_group: str = None) -> dict:
        """
        Internal control formatting plots. Can be used to change plot filling mode for pairs of indicators when.

        :param str indicator_column: column name
        :param my_axis_group: Fill mode for indicator. Color can be forced to fill to zero line with "tozeroy" or between two indicators
         in same axis group with "tonexty".
        :return dict: columns with its assigned fill mode.

        """
        if my_axis_group:
            try:
                assert my_axis_group[0] == 'y' and my_axis_group[1:].isnumeric()
            except Exception:
                print(f"Axis group name need to be y, y2, y3, etc")
                return self.indicators_filled_mode
        if indicator_column and my_axis_group:
            self.axis_groups.update({indicator_column: my_axis_group})
        return self.axis_groups

    def set_plot_splitted_serie_couple(self, indicator_column_up: str = None, indicator_column_down: str = None, splitted_dfs: list = None,
                                       color_up: str = 'rgba(35, 152, 33, 0.5)', color_down: str = 'rgba(245, 63, 39, 0.5)') -> dict:
        """
        Modify the control for splitted series in plots with colored area in two colors by relative position.

        If no params passed, then returns dict actual contents.

        :param str indicator_column_up: An existing column from a BinPan Symbol's class dataframe to plot as up serie (green color).
        :param str indicator_column_down: An existing column from a BinPan Symbol's class dataframe to plot as down serie (red clor).
        :param tuple splitted_dfs: A list of pairs of a splitted dataframe by two columns.
        :param color_up: An rgba formatted color: https://rgbacolorpicker.com/
        :param color_down: An rgba formatted color: https://rgbacolorpicker.com/
        :return dict: A dictionary with auxiliar data about plot areas with two colours by relative position.

        """
        if indicator_column_up and indicator_column_down and splitted_dfs:
            self.plot_splitted_serie_couples.update({
                indicator_column_down: [indicator_column_up, indicator_column_down, splitted_dfs, color_up, color_down]})
        return self.plot_splitted_serie_couples

    def remove_plot_info_associated_columns(self, columns: list, row_level: str) -> list:
        """
        Completely remove plot info for a column in main dataframe of klines.

        :param list columns: List of columns to be deleted.
        :param str row_level: Position of columns to be deleted for a row.
        """
        associated_columns = columns
        if row_level != 1:
            associated_columns += list(set([k for k, v in self.row_control.items() if v == row_level]))
        binpan_logger.info(f"Removing plot info for associated columns: {associated_columns}")
        if row_level != 1:
            self.row_counter -= 1
        for c in associated_columns:
            self.remove_plot_info_for_column(c)
        return list(set(associated_columns))

    def remove_plot_info_for_column(self, column: str):
        """
        Remove plot info for a column in main dataframe of klines.
        :param column:
        """
        binpan_logger.info(f"Removing plot info for column {column}")
        if column in self.row_control:
            del self.row_control[column]
        if column in self.color_control:
            del self.color_control[column]
        if column in self.color_fill_control:
            del self.color_fill_control[column]
        if column in self.indicators_filled_mode:
            del self.indicators_filled_mode[column]
        if column in self.axis_groups:
            del self.axis_groups[column]
        if column in self.plot_splitted_serie_couples:
            del self.plot_splitted_serie_couples[column]

    def plot(self,
             width: int = 1800,
             height: int = 1000,
             candles_ta_height_ratio: float = 0.75,
             volume: bool = True,
             title: str = None,
             yaxis_title: str = 'Price',
             overlapped_indicators: list = None,
             priced_actions_col: str = 'Close',
             actions_col: str = None,
             marker_labels: dict = None,
             markers: list = None,
             marker_colors: list = None,
             background_color=None,
             zoom_start_idx=None,
             zoom_end_idx=None,
             support_lines: list = None,
             support_lines_color: str = 'darkblue',
             resistance_lines: list = None,
             resistance_lines_color: str = 'darkred',
             date: str = None,
             date_radio: int = 20):
        """
        Plots a candles figure for the object.

        Also plots any other technical indicator grabbed.

        .. image:: images/candles.png
           :width: 1000
           :alt: Candles with some indicators

        :param int width: Width of the plot.
        :param int height: Height of the plot.
        :param float candles_ta_height_ratio: Proportion between candles and the other indicators. Not considering overlap ones
            in the candles plot.
        :param bool volume: Plots volume.
        :param str title: A tittle for the plot.
        :param str yaxis_title: A title for the y axis.
        :param list overlapped_indicators: Can declare as overlap in the candles plot some column.
        :param str priced_actions_col: Priced actions to plot annotations over the candles, like buy, sell, etc. Under developing.
        :param str actions_col: A column containing actions like buy or sell. Under developing.
        :param dict marker_labels: Names for the annotations instead of the price. For 'buy' tags and 'sell' tags.
         Default is {'buy': 1, 'sell': -1}
        :param list markers: Plotly marker type. Usually, if referenced by number will be a not filled mark and using string name will be
            a color filled one. Check plotly info: https://plotly.com/python/marker-style/
        :param list marker_colors: Colors of the annotations.
        :param str background_color: Sets background color. Select a valid plotly color name.
        :param int zoom_start_idx: It can zoom to an index interval.
        :param int zoom_end_idx: It can zoom to an index interval.
        :param list support_lines: A list of prices to plot horizontal lines in the candles plot for supports or any other level.
        :param str support_lines_color: A color for horizontal lines, 'darkblue' is by default.
        :param list resistance_lines: A list of prices to plot horizontal lines in the candles plot for resistances or any other level.
        :param str resistance_lines_color: A color for horizontal lines, 'darkred' is by default.
        :param str date: A date in string format to plot a zoom of a radio of klines up and down in time. Useful to inspect dates.
             Incompatible with zoom_start_idx and zoom_end_idx. Strings formatted as "2022-05-11 06:45:42"
        :param str date_radio: A radio in klines to plot a zoom around a date.
        """
        if not overlapped_indicators:
            overlapped_indicators = []
        assert not ((zoom_start_idx or zoom_end_idx) and date), "zoom_start_idx or zoom_end_idx and date are incompatible"

        if not date:
            temp_df = self.df.iloc[zoom_start_idx:zoom_end_idx]
        else:
            date_ms = convert_str_date_to_ms(date=date, time_zone=self.time_zone)
            tick_ms = tick_seconds[self.tick_interval] * 1000
            start_radio = date_ms - (tick_ms * date_radio)
            end_radio = date_ms + (tick_ms * date_radio)
            temp_df = self.df.loc[(self.df['Open timestamp'] >= start_radio) & (self.df['Open timestamp'] <= end_radio)]

        if not title:
            title = temp_df.index.name
            has_sr = support_lines or resistance_lines or any(c for c in temp_df.columns if c.startswith(("Support", "Resistance")))
            if has_sr and self.sr_data_source:
                quality_map = {
                    "atomic trades": "alta precisión",
                    "aggregated trades": "precisión media",
                    "klines": "precisión aproximada",
                }
                quality = quality_map.get(self.sr_data_source, self.sr_data_source)
                title += f" — S/R desde {self.sr_data_source} ({quality})"

        indicators_series = [temp_df[k] for k in self.row_control.keys()]
        indicator_names = [temp_df[k].name for k in self.row_control.keys()]
        indicators_colors = [self.color_control[k] for k in self.row_control.keys()]
        indicators_colors = [c if type(c) == str else _plotting().plotly_colors[c] for c in indicators_colors]

        rows_pos = [self.row_control[k] for k in self.row_control.keys()]

        if support_lines:
            for s_value in support_lines:
                overlapped_indicators += [pd.Series(index=temp_df.index, data=s_value)]
                indicator_names += [f"Support {s_value}"]
                indicators_colors += [support_lines_color]
        if resistance_lines:
            for r_value in resistance_lines:
                overlapped_indicators += [pd.Series(index=temp_df.index, data=r_value)]
                indicator_names += [f"Resistance {r_value}"]
                indicators_colors += [resistance_lines_color]

        if zoom_start_idx is not None or zoom_end_idx is not None:
            zoomed_plot_splitted_serie_couples = zoom_cloud_indicators(self.plot_splitted_serie_couples, main_index=list(self.df.index),
                                                                       start_idx=zoom_start_idx, end_idx=zoom_end_idx)
        else:
            zoomed_plot_splitted_serie_couples = self.plot_splitted_serie_couples
        return _plotting().candles_tagged(data=temp_df,
                              width=width,
                              height=height,
                              candles_ta_height_ratio=candles_ta_height_ratio,
                              plot_volume=volume,
                              title=title,
                              yaxis_title=yaxis_title,
                              on_candles_indicator=overlapped_indicators,
                              priced_actions_col=priced_actions_col,
                              actions_col=actions_col,
                              indicator_series=indicators_series,
                              indicator_names=indicator_names,
                              indicator_colors=indicators_colors,
                              fill_control=self.color_fill_control,
                              indicators_filled_mode=self.indicators_filled_mode,
                              axis_groups=self.axis_groups,
                              plot_splitted_serie_couple=zoomed_plot_splitted_serie_couples,
                              rows_pos=rows_pos,
                              markers_labels=marker_labels,
                              plot_bgcolor=background_color,
                              markers=markers,
                              marker_colors=marker_colors,
                              red_timestamps=self.red_timestamps,
                              blue_timestamps=self.blue_timestamps)

    def plot_agg_trades_size(self, max_size: int = 60, height: int = 1000, logarithmic: bool = False, overlap_prices: bool = True,
                             group_big_data: int = None, shifted: int = 1, title: str = None):
        """
        It plots a time series graph plotting aggregated trades sized by quantity and color if taker or maker buyer.

        Can be used with trades (requieres calling for trades before, or using candles and volume from the object to avoid
        waiting long time intervals grabbing the trades.)

        It can be useful finding support and resistance zones.

        .. image:: images/plot_trades_size.png
           :width: 1000
           :alt: Candles with some indicators

        :param int max_size: Max size for the markers. Default is 60. Useful to show whales operating.
        :param int height: Default is 1000.
        :param bool logarithmic: If logarithmic, then "y" axis scale is shown in logarithmic scale.
        :param int group_big_data: If true, groups data in height bins, this can get faster plotting for big quantity of trades.
        :param bool shifted: If True, shifts prices to plot klines one step to the right, that's more natural to see trades action in price.
        :param bool overlap_prices: If True, plots overlap line with High and Low prices.
        :param title: Graph title

        """
        if self.agg_trades.empty:
            binpan_logger.info(empty_agg_trades_msg)
            return
        if not title:
            title = f"Size aggregated trade categories {self.symbol}"
        managed_data = self.agg_trades.copy(deep=True)

        if overlap_prices:
            overlap_prices = self.df

        if not group_big_data:
            return _plotting().plot_trades(data=managed_data,
                               max_size=max_size,
                               height=height,
                               logarithmic=logarithmic,
                               overlap_prices=overlap_prices,
                               shifted=shifted,
                               title=title)
        else:
            return _plotting().plot_trades(data=managed_data,
                               max_size=max_size,
                               height=height,
                               logarithmic=logarithmic,
                               overlap_prices=overlap_prices,
                               shifted=shifted,
                               title=title)

    def plot_atomic_trades_size(self, max_size: int = 60, height: int = 1000, logarithmic: bool = False, overlap_prices: bool = True,
                                group_big_data: int = None, shifted: int = 1, title: str = None):
        """
        It plots a time series graph plotting atomic trades sized by quantity and color if taker or maker buyer.

        Can be used with trades (requieres calling for trades before, or using candles and volume from the object to avoid
        waiting long time intervals grabbing the trades.)

        It can be useful finding support and resistance zones.

        :param int max_size: Max size for the markers. Default is 60. Useful to show whales operating.
        :param int height: Default is 1000.
        :param bool logarithmic: If logarithmic, then "y" axis scale is shown in logarithmic scale.
        :param int group_big_data: If true, groups data in height bins, this can get faster plotting for big quantity of trades.
        :param bool shifted: If True, shifts prices to plot klines one step to the right, that's more natural to see trades action in price.
        :param bool overlap_prices: If True, plots overlap line with High and Low prices.
        :param title: Graph title

        .. image:: images/plot_trades_size_log.png
           :width: 800
           :alt: Atomic trades size bubble chart

        """
        if self.atomic_trades.empty:
            binpan_logger.info(empty_atomic_trades_msg)
            return
        if not title:
            title = f"Size atomic trade categories {self.symbol}"
        managed_data = self.atomic_trades.copy(deep=True)

        if overlap_prices:
            overlap_prices = self.df

        if not group_big_data:
            return _plotting().plot_trades(data=managed_data,
                               max_size=max_size,
                               height=height,
                               logarithmic=logarithmic,
                               overlap_prices=overlap_prices,
                               shifted=shifted,
                               title=title)
        else:
            return _plotting().plot_trades(data=managed_data,
                               max_size=max_size,
                               height=height,
                               logarithmic=logarithmic,
                               overlap_prices=overlap_prices,
                               shifted=shifted,
                               title=title)

    def plot_reversal(self, min_height: int = None, min_reversal: int = None, text_index: bool = True, from_atomic: bool = False, **kwargs):
        """
        Plots reversal candles. It requires aggregated or atomic trades fetched previously.

        BinPan manages aggregated trades from binance API.

        :param int min_height: It defaults to previous set. Can be reset when charts.
        :param min_reversal: It defaults to previous set. Can be reset when charts.
        :param bool text_index: If True, plots klines equally spaced. This allows to plot volume.
        :param bool from_atomic: If True, klines are obtained from atomic trades.
        :return:

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
        if not from_atomic and self.agg_trades.empty:
            binpan_logger.info(empty_agg_trades_msg)
            return
        if from_atomic and self.atomic_trades.empty:
            binpan_logger.info(empty_atomic_trades_msg)
            return

        if min_height:
            self.min_height = min_height
        if min_reversal:
            self.min_reversal = min_reversal

        if min_height or min_reversal:
            if from_atomic:
                self.reversal_atomic_klines = self.get_reversal_atomic_candles(min_height=self.min_height, min_reversal=self.min_reversal)
            else:
                self.reversal_agg_klines = self.get_reversal_agg_candles(min_height=self.min_height, min_reversal=self.min_reversal)

        if not 'title' in kwargs.keys():
            source = "atomic trades (alta precisión)" if from_atomic else "aggregated trades (precisión media)"
            kwargs['title'] = f"Reversal Candles {self.min_height}/{self.min_reversal} {self.symbol} — desde {source}"
        if not 'yaxis_title' in kwargs.keys():
            kwargs['yaxis_title'] = f"Price {self.symbol}"
        if not 'candles_ta_height_ratio' in kwargs.keys():
            kwargs['candles_ta_height_ratio'] = 0.7

        if from_atomic:
            return _plotting().candles_ta(data=self.reversal_atomic_klines, plot_volume='Quantity', text_index=text_index,
                              volume_window=self.plotting_volume_ma, **kwargs)
        else:
            return _plotting().candles_ta(data=self.reversal_agg_klines, plot_volume='Quantity', text_index=text_index,
                              volume_window=self.plotting_volume_ma, **kwargs)

    def set_plotting_volume_ma(self, window: int = 21) -> None:
        """
        Set a window for plotting volume moving average on the candles plot when volumen bars are plotted.

        :param int window: A window for the moving average.
        """
        self.plotting_volume_ma = window
        binpan_logger.info(f"Plotting volume moving average set to {window}")

    def plot_trades_pie(self, categories: int = 25, logarithmic=True, title: str = None):
        """
        Plots a pie chart. Useful profiling size of trades. Size can be distributed in a logarithmic scale.

        .. image:: images/plot_trades_pie.png
           :width: 1000
           :alt: Candles with some indicators

        :param categories: How many groups of sizes.
        :param logarithmic: Logarithmic scale to show more small sizes.
        :param title: A title for the plot.

        """
        if self.agg_trades.empty:
            binpan_logger.info(empty_agg_trades_msg)
            return
        if not title:
            title = f"Size trade categories {self.symbol}"
        return _plotting().plot_pie(serie=self.agg_trades['Quantity'], categories=categories, logarithmic=logarithmic, title=title)

    def plot_aggression_sizes(self, bins=50, hist_funct='sum', height=900, from_trades=False, title: str = None,
                              total_volume_column: str = None, partial_vol_column: str = None, **kwargs_update_layout):
        """
        Binance fees can be cheaper for maker orders, many times when big traders, like whales, are operating . Showing what are doing
        makers.

        It shows which kind of volume or trades came from, aggressive_sellers or aggressive_byers.

        Can be useful finding support and resistance zones.

        .. image:: images/makers_vs_takers_plot.png
           :width: 1000
           :alt: Candles with some indicators


        :param bins: How many bars.
        :param hist_funct: The way graph data is showed. It can be 'mean', 'sum', 'percent', 'probability', 'density', or 'probability
         density'
        :param height: Height of the graph.
        :param from_trades: Requieres grabbing trades before.
        :param title: A title.
        :param total_volume_column: The column with the total volume. It defaults automatically.
        :param partial_vol_column: The column with the partial volume. It defaults automatically. API shows maker or taker separated
         volumes.
        :param kwargs_update_layout: Optional

        """
        if from_trades or not self.agg_trades.empty:
            if self.agg_trades.empty:
                binpan_logger.info(empty_agg_trades_msg)
                return
            else:
                _df = self.agg_trades.copy(deep=True)

                if not total_volume_column:
                    total_volume_column = 'Quantity'

                if not partial_vol_column:
                    partial_vol_column = 'Buyer was maker'

                aggressive_sellers = _df.loc[_df[partial_vol_column]][total_volume_column]
                aggressive_byers = _df.loc[~_df[partial_vol_column]][total_volume_column]

        else:
            _df = self.df.copy()
            if not total_volume_column:
                total_volume_column = 'Volume'
            if not partial_vol_column:
                partial_vol_column = 'Taker buy base volume'
            aggressive_sellers = _df[partial_vol_column]
            aggressive_byers = _df[total_volume_column] - aggressive_sellers

        if not title:
            title = f"Histogram for sizes in aggressive sellers vs aggressive byers {self.symbol} ({hist_funct})"

        return _plotting().plot_hists_vs(x0=aggressive_sellers, x1=aggressive_byers, x0_name="Taker seller", x1_name='Taker buyer',
                             bins=bins, hist_funct=hist_funct, height=height, title=title, **kwargs_update_layout)

    def plot_market_profile(self, bins: int = 100, hours: int = None, minutes: int = None, startTime: int | str = None,
                            endTime: int | str = None, height=900, from_agg_trades=False, from_atomic_trades=False, title: str = None,
                            time_zone: str = None, **kwargs_update_layout):
        """
        Plots volume histogram by prices segregated aggressive buyers from sellers.


        :param int bins: How many bars.
        :param int hours: If passed, it use just last passed hours for the plot.
        :param int minutes: If passed, it use just last passed minutes for the plot.
        :param startTime: If passed, it use just from the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :type startTime: int | str
        :param endTime: If passed, it use just until the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :type endTime: int | str
        :param height: Height of the graph.
        :param from_agg_trades: Requieres grabbing aggregated trades before.
        :param from_atomic_trades: Requieres grabbing atomic trades before.
        :param title: A title.
        :param str time_zone: A time zone for time index conversion. Example: "Europe/Madrid"
        :param kwargs_update_layout: Optional

        .. image:: images/plotting/market_profile.png
            :width: 1000

        """
        try:
            assert not (from_agg_trades and from_atomic_trades)
        except AssertionError:
            raise BinPanException(f"Please specify just one source of data, atomic trades or aggregated, not both.")

        if not title:
            title = f"Market profile {self.symbol}"
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
            title += ' — desde aggregated trades (precisión media)'
            _df = self.agg_trades.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            return _plotting().bar_plot(df=_df, x_col_to_bars='Price', y_col='Quantity', bar_segments='Buyer was maker', split_colors=True,
                            bins=bins, title=title, height=height, y_axis_title='Buy takers VS Buy makers', horizontal_bars=True,
                            **kwargs_update_layout)
        elif from_atomic_trades:
            title += ' — desde atomic trades (alta precisión)'
            _df = self.atomic_trades.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            return _plotting().bar_plot(df=_df, x_col_to_bars='Price', y_col='Quantity', bar_segments='Buyer was maker', split_colors=True,
                            bins=bins, title=title, height=height, y_axis_title='Buy takers VS Buy makers', horizontal_bars=True,
                            **kwargs_update_layout)
        else:
            title += ' — desde klines (precisión aproximada)'
            _df = self.df.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            binpan_logger.info(f"Using klines data. For deeper info add trades data, example: my_symbol.get_agg_trades()")
            profile = market_profile_from_klines_melt(df=_df)
            profile.reset_index(inplace=True)
            return _plotting().bar_plot(df=profile, x_col_to_bars='Market_Profile', y_col='Volume', bar_segments='Is_Maker', split_colors=True,
                            bins=bins, title=title + " from klines", height=height, y_axis_title='Buy takers VS Buy makers',
                            horizontal_bars=True,
                            **kwargs_update_layout)

    def plot_trades_scatter(self, x: str = None, y: str = None, dot_symbol='Buyer was maker', color: str = None, marginal=True,
                            from_trades=True, height=1000, color_referenced_to_y=True,
                            # useful to compare volume with taker volume for coloring
                            **kwargs):
        """
        A scatter plot showing each price level volume or trades.

        It can be useful finding support and resistance zones.

        .. image:: images/plot_trades_scatter.png
           :width: 1000
           :alt: Candles with some indicators


        :param dot_symbol: Column with discrete values to assign different symbols for the plot marks.
        :param x: Name of the column with prices. From trades or candles.
        :param y: Name of the column with sizes. From trades or candles.
        :param color: Column with values to use in color scale.
        :param marginal: Show or not lateral plots.
        :param from_trades: Uses trades instead of candles. Useful to avoid grabbing very long time intervals. Result should be similar.
        :param height: Height of the plot.
        :param color_referenced_to_y: Scales color in y axis.
        :param kwargs: Optional plotly args.

        """
        if not x:
            x = ['Price', 'Close']
        if not y:
            y = ['Quantity', 'Volume']
        if not color:
            color = ['Buyer was maker', 'Taker buy base volume']
        if self.agg_trades.empty and from_trades:
            binpan_logger.info(empty_agg_trades_msg)
            return
        if not from_trades:
            data = self.df.copy(deep=True)
            if not (type(x) == str and type(y) == str and type(color) == str):
                x = x[1]
                y = y[1]
                if color_referenced_to_y:
                    color = data[color[1]] / data[y]
                    # kwargs.update({'hover_data': color})
                    kwargs.update({'labels': {"color": "Maker buyer volume / Total volume"}})
            title = f"Priced volume for {self.symbol} data obtained from volume and candlesticks."
            return _plotting().plot_scatter(df=data, x_col=x, y_col=y, color=color, marginal=marginal, title=title, height=height, **kwargs)
        else:
            data = self.agg_trades.copy(deep=True)
            if not (type(x) == str and type(y) == str) and type(color):
                x = x[0]
                y = y[0]
                color = color[0]
            title = f"Priced volume for {self.symbol} data obtained from historical trades."
            return _plotting().plot_scatter(df=data, x_col=x, y_col=y, symbol=dot_symbol, color=color, marginal=marginal, title=title, height=height,
                                **kwargs)

    def plot_orderbook(self, accumulated=True, title='Depth orderbook plot', height=800, plot_y="Quantity", **kwargs):
        """
        Plots orderbook depth.

        .. image:: images/plot_orderbook.png
           :width: 800
           :alt: Order book depth chart

        """
        if self.orderbook.empty:
            binpan_logger.info("Orderbook not downloaded. Please add orderbook data with: my_binpan.get_orderbook()")
            return
        return _plotting().orderbook_depth(df=self.orderbook, accumulated=accumulated, title=title, height=height, plot_y=plot_y, **kwargs)

    def plot_orderbook_density(self, x_col="Price", color='Side', bins=300, histnorm: str = 'density', height: int = 800, title: str = None,
                               **update_layout_kwargs):
        """
        Plot a distribution plot for a dataframe column. Plots line for kernel distribution.

        :param str x_col: Column name for x-axis data.
        :param str color: Column name with tags or any values for using as color scale.
        :param int bins: Columns in histogram.
        :param str histnorm: One of 'percent', 'probability', 'density', or 'probability density' from plotly express documentation.
            https://plotly.github.io/plotly.py-docs/generated/plotly.express.histogram.html
        :param int height: Plot sizing.
        :param str title: A title string

        .. image:: images/orderbook_density.png
           :width: 800
           :alt: Order book density plot

        """

        if self.orderbook.empty:
            binpan_logger.info("Orderbook not downloaded. Please add orderbook data with: my_binpan.get_orderbook()")
            return

        if not title:
            title = f"Distribution plot for order book {self.symbol}"

        return _plotting().dist_plot(df=self.orderbook, x_col=x_col, color=color, bins=bins, histnorm=histnorm, height=height, title=title,
                         **update_layout_kwargs)

    def plot_taker_maker_ratio_profile(self, bins: int = 100, hours: int = None, minutes: int = None, startTime: int | str = None,
                                       endTime: int | str = None, from_agg_trades=False, from_atomic_trades=False, time_zone: str = None,
                                       title: str = "Taker Buy Ratio Profile", height=1200, width=800, **kwargs_update_layout):
        """
        Plots taker vs maker ratio profile.

        :param int bins: How many bars.
        :param int hours: If passed, it use just last passed hours for the plot.
        :param int minutes: If passed, it use just last passed minutes for the plot.
        :param startTime: If passed, it use just from the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :type startTime: int | str
        :param endTime: If passed, it use just until the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :type endTime: int | str
        :param height: Height of the graph.
        :param width: Width of the graph.
        :param from_agg_trades: Requieres grabbing aggregated trades before.
        :param from_atomic_trades: Requieres grabbing atomic trades before.
        :param title: A title.
        :param str time_zone: A time zone for time index conversion. Example: "Europe/Madrid"
        :param kwargs_update_layout: Optional

        .. image:: images/plotting/taker_ratio_profile.png
            :width: 1000

        """
        if title == "Taker Buy Ratio Profile":
            if from_atomic_trades:
                title += " — desde atomic trades (alta precisión)"
            elif from_agg_trades:
                title += " — desde aggregated trades (precisión media)"
            else:
                title += " — desde klines (precisión aproximada)"
        profile = self.get_taker_maker_ratio_profile(bins=bins, hours=hours, minutes=minutes, startTime=startTime, endTime=endTime,
                                                     from_agg_trades=from_agg_trades, from_atomic_trades=from_atomic_trades,
                                                     time_zone=time_zone)
        return _plotting().profile_plot(serie=profile, title=title, height=height, width=width, x_axis_title="Price Buckets",
                            y_axis_title="Taker/Maker ratio", vertical_bar=0.5, **kwargs_update_layout)
