"""

This is the main classes file.

"""

import pandas as pd
import handlers.logs
import handlers.market
import handlers.quest
import handlers.exceptions
import handlers.time_helper
import handlers.plotting
import handlers.wallet
import handlers.files_filters

import pandas_ta as ta
from random import choice
import numpy as np


binpan_logger = handlers.logs.Logs(filename='./logs/binpan.log', name='binpan', info_level='INFO')
tick_seconds = handlers.time_helper.tick_seconds

__version__ = "0.0.3"

plotly_colors = handlers.plotting.plotly_colors


class Symbol(object):
    """
    Creates an object from binance klines and/or trade data. It contains the raw api response and a dataframe that can be modified.

    Any symbol can be used as argument and any time interval like: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h',
    '12h', '1d', '3d', '1w', '1M'

    Object has several plot methods.

    :param str symbol:  It can be any symbol in the binance exchange, like BTCUSDT, ethbusd or any other. Capital letters doesn't matter.

    :param str tick_interval: Any candles interval available in binance. Capital letters doesn't matter.

    :param int or str start_time:  It can be an integer in milliseconds from epoch (1970-01-01 00:00:00 UTC) or any string in the formats:

        - %Y-%m-%d %H:%M:%S.%f:       **2022-05-11 06:45:42.124567**
        - %Y-%m-%d %H:%M:%S:          **2022-05-11 06:45:42**

       If start time is passed, it gets the next open according to the tick interval selected except an exact open time passed.

    :param int or str end_time:    It can be an integer in milliseconds from epoch (1970-01-01 00:00:00 UTC) or any string in the formats:

        - %Y-%m-%d %H:%M:%S.%f:       **2022-05-11 06:45:42.124**
        - %Y-%m-%d %H:%M:%S:          **2022-05-11 06:45:42**

       If end time is passed, it gets candles till the previous close according to the selected tick interval.

       Example with daily time saving inside the interval:

        .. code-block:: python

            import binpan

            btcusdt = binpan.Candles(symbol='btcusdt',
                                     tick_interval='15m',
                                     time_zone='Europe/Madrid',
                                     start_time='2021-10-31 01:00:00',
                                     end_time='2021-10-31 03:00:00')

            btcusdt.candles

                                            Open	        High	        Low	        Close	        Volume
            BTCUSDT 15m Europe/Madrid

            2021-10-31 01:00:00+02:00	61540.32	61780.44	61488.35	61737.39	316.93504
            2021-10-31 01:15:00+02:00	61737.39	61748.13	61486.86	61637.99	485.50681
            2021-10-31 01:30:00+02:00	61637.99	61777.71	61479.76	61743.32	441.29718
            2021-10-31 01:45:00+02:00	61743.31	61900.00	61640.55	61859.19	505.87893
            2021-10-31 02:00:00+02:00	61859.19	62328.38	61859.18	62328.13	1046.30598
            2021-10-31 02:15:00+02:00	62328.13	62399.00	62142.74	62246.30	679.96187
            2021-10-31 02:30:00+02:00	62246.30	62371.72	62171.01	62301.39	355.25273
            2021-10-31 02:45:00+02:00	62301.39	62400.15	62220.27	62375.29	568.02797
            2021-10-31 02:00:00+01:00	62375.28	62405.30	62260.01	62367.79	421.52566
            2021-10-31 02:15:00+01:00	62367.79	62394.06	62219.50	62270.00	537.53843
            2021-10-31 02:30:00+01:00	62270.00	62371.49	62156.83	62169.01	452.62728
            2021-10-31 02:45:00+01:00	62169.01	62227.00	62114.18	62169.00	283.45948


    :param int limit:       The limit is the quantity of candles requested. Can combine with start_time or end_time:

                            - If no start or end timestamp passed, gets current timestamp and limit of candles backwards.
                            - If just start time or endtime passed, then applies the limit to find the endtime
                              or start time respectively.
                            - if endtime and start time passed, it gets all the ticks between, even if more than 1000 by calling
                              several times to API until completed. Limit ignored.

    :param str time_zone:   The index of the pandas dataframe in the object can be converted to any timezone, i.e. "Europe/Madrid"

                           - TZ database: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

                        The binance API use milliseconds from epoch (1970-01-01 00:00:00 UTC), that means that converting to a timezoned
                        date with daily time saving changes, the result would show the change in the timestamp.

                        Candles are ordered with the timestamp regardless of the index name, even if the index shows the hourly change
                        because daily time saving changes.

    :param bool time_index:  Shows human-readable index in the dataframe. Set to False shows numeric index. default is True.

    :param bool closed:      The last candle is a closed one in the moment of the creation, instead of a running candle not closed yet.

    :param int display_columns:     Number of columns in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param int display_rows:        Number of rows in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param int display_width:       Display width in the dataframe display. Convenient to adjust in jupyter notebooks.

    Examples:

    .. code-block::

        binpan_object = binpan.Candles(symbol='ethbusd',
                                       tick_interval='5m',
                                       limit=100,
                                       display_rows=10)
        binpan_object.df()

                                        Open    High    Low     Close   Volume          Quote volume    Trades  Taker buy base volume   Taker buy quote volume

        ETHBUSD 5m UTC

        2022-06-22 08:55:00+00:00	1077.29	1081.11	1076.34	1080.18	810.1562	8.737861e+05	1401.0	410.0252	4.422960e+05
        2022-06-22 09:00:00+00:00	1080.28	1087.31	1079.90	1085.03	2446.4199	2.653134e+06	3391.0	1253.6380	1.359514e+06
        2022-06-22 09:05:00+00:00	1085.14	1085.98	1080.70	1080.99	1502.4415	1.628110e+06	1664.0	509.9937	5.525626e+05
        2022-06-22 09:10:00+00:00	1080.79	1084.53	1080.26	1083.30	906.5092	9.814984e+05	1438.0	514.6718	5.573028e+05
        2022-06-22 09:15:00+00:00	1083.42	1085.35	1081.41	1081.53	1191.9566	1.291633e+06	1549.0	736.5743	7.983178e+05
        ...	...	...	...	...	...	...	...	...	...
        2022-06-22 16:50:00+00:00	1067.61	1070.00	1066.21	1069.32	1951.1378	2.084288e+06	1535.0	1396.4956	1.491744e+06
        2022-06-22 16:55:00+00:00	1069.47	1078.61	1069.47	1077.78	2363.1639	2.539150e+06	3578.0	1239.7735	1.332085e+06
        2022-06-22 17:00:00+00:00	1077.96	1079.32	1071.25	1073.00	1452.3712	1.560748e+06	2534.0	651.9175	7.006751e+05
        2022-06-22 17:05:00+00:00	1072.76	1075.04	1071.00	1073.47	728.9761	7.820787e+05	1383.0	303.3896	3.254396e+05
        2022-06-22 17:10:00+00:00	1073.42	1076.30	1073.24	1075.77	228.4269	2.456515e+05	426.0	127.7424	1.373634e+05
        100 rows × 9 columns

    Created objects contain different data like:
    - mysymbol.df: shows candles dataframe
    - mysymbol.trades: shows aggregated trades, if requested. This is optional and can be added anytime.
    """

    def __init__(self,
                 symbol: str,
                 tick_interval: str,
                 start_time: int or str = None,
                 end_time: int or str = None,
                 limit: int = 1000,
                 time_zone: str = 'UTC',
                 time_index: bool = True,
                 closed: bool = True,
                 display_columns=25,
                 display_rows=10,
                 display_max_rows=25,
                 display_width=320):

        # check correct tick interval passed
        tick_interval = handlers.time_helper.check_tick_interval(tick_interval)

        self.original_candles_cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote volume',
                                      'Trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore']
        self.presentation_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote volume', 'Trades', 'Taker buy base volume',
                                     'Taker buy quote volume']

        self.trades_columns = {'M': 'Best price match',
                               'm': 'Buyer was maker',
                               'T': 'Timestamp',
                               'l': 'Last tradeId',
                               'f': 'First tradeId',
                               'q': 'Quantity',
                               'p': 'Price',
                               'a': 'Aggregate tradeId'}

        self.time_cols = ['Open time', 'Close time']
        self.dts_time_cols = ['Open timestamp', 'Close timestamp']

        self.version = __version__
        self.symbol = symbol.upper()
        self.fees = self.get_fees(symbol=self.symbol)
        self.tick_interval = tick_interval
        self.start_time = start_time
        self.end_time = end_time
        self.limit = limit
        self.time_zone = time_zone
        self.time_index = time_index
        self.closed = closed
        self.start_ms_time = None
        self.end_ms_time = None
        self.display_columns = display_columns
        self.display_rows = display_rows
        self.display_max_rows = display_max_rows
        self.display_width = display_width
        self.trades = pd.DataFrame(columns=list(self.trades_columns.values()))
        self.row_control = {}
        self.color_control = {}
        self.row_counter = 1

        self.set_display_columns()
        self.set_display_width()
        self.set_display_rows()
        self.set_display_max_rows()

        binpan_logger.info(f"New instance of CandlesManager {self.version}: {self.symbol}, {self.tick_interval}, limit={self.limit},"
                           f" start={self.start_time}, end={self.end_time}, {self.time_zone}, time_index={self.time_index}"
                           f", closed_candles={self.closed}")

        if type(self.start_time) == str:
            self.start_time = handlers.time_helper.convert_string_to_milliseconds(self.start_time, timezoned=self.time_zone)

        if type(self.end_time) == str:
            self.end_time = handlers.time_helper.convert_string_to_milliseconds(self.end_time, timezoned=self.time_zone)

        # check for limit quantity for big queries
        self.start_theoretical, self.end_theoretical = handlers.time_helper.time_interval(tick_interval=self.tick_interval,
                                                                                          limit=self.limit,
                                                                                          start=self.start_time,
                                                                                          end=self.end_time)
        if self.closed:
            now = handlers.time_helper.utc()
            current_open = handlers.time_helper.open_from_milliseconds(ms=now, tick_interval=self.tick_interval)
            if self.end_theoretical >= current_open:
                self.end_theoretical = current_open - 1000  # resta para solicitar la vela anterior, que está cerrada

        self.ticks = handlers.time_helper.ticks_between_timestamps(start=self.start_theoretical,
                                                                   end=self.end_theoretical,
                                                                   tick_interval=self.tick_interval)
        # loop big queries
        if self.ticks > 1000:
            raw_candles = []
            start_pointer = self.start_theoretical
            end_pointer = handlers.time_helper.open_from_milliseconds(ms=self.end_theoretical, tick_interval=self.tick_interval)

            while start_pointer <= self.end_theoretical:

                response = handlers.market.get_candles_from_start_time(start_time=start_pointer,
                                                                       symbol=self.symbol,
                                                                       tick_interval=self.tick_interval,
                                                                       limit=1000)
                raw_candles = raw_candles + response
                last_raw_open_ts = raw_candles[-1][0]  # mira el open
                if last_raw_open_ts >= end_pointer:
                    break
                start_pointer = int(response[-1][0]) + (tick_seconds[tick_interval] * 1000)

            # descarta sobrantes
            overtime_candle_ts = handlers.time_helper.next_open_by_milliseconds(ms=end_pointer, tick_interval=self.tick_interval)
            raw_candles = [i for i in raw_candles if int(i[0]) < overtime_candle_ts]

        elif not end_time and not start_time:
            raw_candles = handlers.market.get_last_candles(symbol=self.symbol,
                                                           tick_interval=tick_interval,
                                                           limit=limit)
        else:
            raw_candles = handlers.market.get_candles_by_time_stamps(start_time=self.start_time,
                                                                     end_time=self.end_time,
                                                                     symbol=self.symbol,
                                                                     tick_interval=self.tick_interval,
                                                                     limit=self.limit)
        if self.closed:
            raw_candles = raw_candles[:-1]

        self.raw = raw_candles

        dataframe = self.parse_candles_to_dataframe(response=self.raw,
                                                    columns=self.original_candles_cols,
                                                    time_cols=self.time_cols,
                                                    symbol=self.symbol,
                                                    tick_interval=self.tick_interval,
                                                    time_zone=self.time_zone,
                                                    time_index=self.time_index)
        self.df = dataframe
        # self.candles = dataframe[self.presentation_columns]
        self.len = len(self.df)

    def __repr__(self):
        return str(self.df)

    def set_display_columns(self, display_columns=None):
        """
        Change the number of maximum columns shown in the display of the dataframe.

        :param int display_columns: Integer

        """
        if display_columns:
            pd.set_option('display.max_columns', display_columns)
        else:
            pd.set_option('display.max_columns', self.display_columns)

    def set_display_rows(self, display_rows=None):
        """
        Change the number of minimum rows shown in the display of the dataframe.

        :param int display_rows: Integer

        """
        if display_rows:
            pd.set_option('display.min_rows', display_rows)

        else:
            pd.set_option('display.min_rows', self.display_rows)

    def set_display_max_rows(self, display_max_rows=None):
        """
        Change the number of minimum rows shown in the display of the dataframe.

        :param int display_max_rows: Integer

        """
        if display_max_rows:
            pd.set_option('display.max_rows', display_max_rows)

        else:
            pd.set_option('display.max_rows', self.display_rows)

    def set_display_width(self, display_width: int = None):
        """
        Change the shown width shown in the display of the dataframe.

        :param int display_width: Integer

        """
        if display_width:
            pd.set_option('display.width', display_width)
        else:
            pd.set_option('display.width', self.display_width)

    @staticmethod
    def set_display_decimals(display_decimals: int):
        """
        Change the decimals shown in the dataframe. It changes all the columns decimals.

        :param int display_decimals: Integer

        """
        arg = f'%.{display_decimals}f'
        pd.set_option('display.float_format', lambda x: arg % x)

    def basic(self,
              exceptions: list = None,
              actions_col='actions',
              inplace=False):
        """
        Shows just a basic selection of columns data in the dataframe. Any column can be excepted from been dropped.

        :param list exceptions: Columns names to keep.
        :param str actions_col: Under development. To keep tags for buy or sell actions.
        :param bool inplace: Keep in the object just the basic columns, loosing any other one.
        :return pd.DataFrame: Pandas DataFrame
        """
        if inplace:
            new_candles = self.basic_dataframe(data=self.df, exceptions=exceptions, actions_col=actions_col)
            self.df = new_candles
            return self.df
        else:
            return self.basic_dataframe(data=self.df, exceptions=exceptions, actions_col=actions_col)

    def drop(self, columns=[], inplace=False) -> pd.DataFrame:
        """
        It drops some columns from the dataframe. If columns list not passed, then defaults to the initial columns.

        Can be used when messing with indicators to clean the object.

        :param: list columns: A list with the columns names to drop. If not passed, it defaults to the initial columns that remain
            from when instanced.
        :param: bool inplace: When true, it drops columns in the object. False just returns a copy without that columns and dataframe
            in the object remains.
        :return pd.DataFrame: Pandas DataFrame with columns dropped.

        """
        if not columns:
            columns = []
            for col in self.df.columns:
                if not col in self.original_candles_cols:
                    columns.append(col)
        try:
            if inplace:
                self.df.drop(columns, axis=1, inplace=True)
                return self.df
            else:
                return self.df.drop(columns, axis=1, inplace=False)

        except KeyError:
            wrong = (set(self.df.columns) | set(columns)) - set(self.df.columns)
            msg = f"BinPan Exception: Wrong column names to drop: {wrong}"
            binpan_logger.error(msg)
            raise Exception(msg)

    def hk(self, inplace=False):
        """
        It computes Heikin Ashi candles. Any existing indicator column will not be recomputed. It is recommended to drop any indicator
         before converting candles to Heikin Ashi.

        :param bool inplace: Change object dataframe permanently whe True is selected. False shows a copy dataframe.
        :return pd.DataFrame: Pandas DataFrame

        """
        df_ = self.df.copy(deep=True)
        cols = ['Open', 'High', 'Low', 'Close']

        heikin_ashi_df = df_.iloc[:][cols]

        heikin_ashi_df['Close'] = (df_['Open'] + df_['High'] + df_['Low'] + df_['Close']) / 4
        for i in range(len(df_)):
            if i == 0:
                heikin_ashi_df.iat[0, 0] = df_['Open'].iloc[0]
            else:
                heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i - 1, 0] + heikin_ashi_df.iat[i - 1, 3]) / 2
        heikin_ashi_df['High'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df_['High']).max(axis=1)
        heikin_ashi_df['Low'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df_['Low']).min(axis=1)
        df_[cols] = heikin_ashi_df[cols]
        df_.index.name = df_.index.name + ' HK'
        if inplace:
            self.df = df_
            return self.df
        else:
            return df_

    def timestamps(self):
        """
        Get the first Open timestamp and the last Close timestamp.

        :return tuple(int, int): Start Open timestamp and end close timestamp

        """
        start = self.df.iloc[0]['Open timestamp']
        end = self.df.iloc[-1]['Close timestamp']
        return start, end

    def get_trades(self):
        """
        Calls the API and creates another dataframe included in the object with the aggregated trades from API for the period of the
         created object.

          .. note::
             If the object covers a long time interval, this action can take a relative long time. The BinPan library take care of the
             API weight and can take a sleep to wait until API weight returns to a low value.

        :return:
        """
        trades = handlers.market.get_historical_aggregated_trades(symbol=self.symbol,
                                                                  startTime=self.start_theoretical,
                                                                  endTime=self.end_theoretical)
        self.trades = self.parse_agg_trades_to_dataframe(response=trades,
                                                         columns=self.trades_columns,
                                                         symbol=self.symbol,
                                                         time_zone=self.time_zone,
                                                         time_index=self.time_index)
        return self.trades

    def plot_rows(self, indicator_column: str = None, row_position: int = None):
        """
        Internal control formatting plots.
        :param indicator_column:
        :param row_position:
        :return:
        """
        if indicator_column and row_position:
            self.row_control.update({indicator_column: row_position})
        return self.row_control

    def plot_colors(self, indicator_column: str = None, color: int or str = None):
        """
        Internal control formatting plots.
        :param indicator_column:
        :param color:
        :return:
        """
        if indicator_column and color:
            if type(color) == int:
                self.color_control.update({indicator_column: plotly_colors[color]})
            elif color in plotly_colors:
                self.color_control.update({indicator_column: color})
        elif indicator_column:
            self.color_control.update({indicator_column: choice(plotly_colors)})
        return self.color_control

    def plot(self,
             width=1800,
             height=1000,
             candles_ta_height_ratio: float = 0.75,
             plot_volume: bool = True,
             title: str = None,
             yaxis_title='Price',
             overlapped_indicators: list = [],
             priced_actions_col='priced_actions',
             actions_col: str = None,
             labels: list = [],
             default_price_for_actions='Close'):
        """
        Plots a candles figure for the object.

        Also plots any other technical indicator grabbed.

        .. image:: images/candles.png
           :width: 1000
           :alt: Candles with some indicators

        :param width: Width of the plot.
        :param height: Height of the plot.
        :param candles_ta_height_ratio: Proportion between candles and the other indicators. Not considering overlay ones
         in the candles plot.
        :param plot_volume: Plots volume.
        :param title: A tittle for the plot.
        :param yaxis_title: A title for the y axis.
        :param overlapped_indicators: Can declare as overlay in the candles plot some column.
        :param priced_actions_col: Priced actions to plot annotations over the candles, like buy, sell, etc. Under developing.
        :param actions_col: A column containing actions like buy or sell. Under developing.
        :param labels: Names for the annotations instead of the price.
        :param default_price_for_actions: Column to use as priced actions in case of not existing an specific prices actions column.
        """

        if not title:
            title = self.df.index.name

        indicators_series = [self.df[k] for k in self.row_control.keys()]
        indicator_names = [self.df[k].name for k in self.row_control.keys()]
        indicators_colors = [self.color_control[k] for k in self.row_control.keys()]
        rows_pos = [self.row_control[k] for k in self.row_control.keys()]

        binpan_logger.debug(f"{indicator_names}\n{indicators_colors}\n{rows_pos}")

        handlers.plotting.candles_tagged(data=self.df,
                                         width=width,
                                         height=height,
                                         candles_ta_height_ratio=candles_ta_height_ratio,
                                         plot_volume=plot_volume,
                                         title=title,
                                         yaxis_title=yaxis_title,
                                         on_candles_indicator=overlapped_indicators,
                                         priced_actions_col=priced_actions_col,
                                         actions_col=actions_col,
                                         indicators_series=indicators_series,
                                         indicator_names=indicator_names,
                                         indicators_colors=indicators_colors,
                                         rows_pos=rows_pos,
                                         labels=labels,
                                         default_price_for_actions=default_price_for_actions)

    def plot_trades_size(self, max_size=60, height=1000, logarithmic=False, title: str = None):
        """
        It plots a time series graph plotting trades sized by quantity and color if taker or maker buyer.

        Can be used with trades (requieres calling for trades before, or using candles and volume from the object to avoid
        waiting long time intervals grabbing the trades.)

        It can be useful finding support and resistance zones.

        .. image:: images/plot_trades_size.png
           :width: 1000
           :alt: Candles with some indicators

        :param int max_size: Max size for the markers. Default is 60. Useful to show whales operating.
        :param int height: Default is 1000.
        :param bool logarithmic: If logarithmic, then "y" axis scale is shown in logarithmic scale.
        :param title: Graph title

        """
        if self.trades.empty:
            binpan_logger.info("Trades not downloaded. Please add trades data with: my_binpan.get_trades()")
            return
        if not title:
            title = f"Size trade categories {self.symbol}"
        handlers.plotting.plot_trade_size(data=self.trades.copy(deep=True),
                                          max_size=max_size,
                                          height=height,
                                          logarithmic=logarithmic,
                                          title=title)

    def plot_trades_pie(self, categories: int = 25, logarithmic=True, title: str = None):
        """
        Plots a pie chart. Useful profiling size of trades. Size can be distributed in a logarithmic scale.

        .. image:: images/plot_trades_pie.png
           :width: 1000
           :alt: Candles with some indicators

        :param categories: How many groups of sizes.
        :param logarithmic: Logaritmic scale to show more small sizes.
        :param title: A title for the plot.

        """
        if self.trades.empty:
            binpan_logger.info("Trades not downloaded. Please add trades data with: my_binpan.get_trades()")
            return
        if not title:
            title = f"Size trade categories {self.symbol}"
        handlers.plotting.plot_pie(serie=self.trades['Quantity'],
                                   categories=categories,
                                   logarithmic=logarithmic,
                                   title=title)

    def makers_vs_takers_plot(self,
                              bins=50,
                              hist_funct='sum',
                              height=900,
                              from_trades=False,
                              title: str = None,
                              total_volume=None,
                              partial_vol=None,
                              **kwargs_update_layout):
        """
        Binance fees can be cheaper for maker orders, many times when big traders, like whales, are operating. Showing what are doing
        makers.

        It shows which kind of volume or trades came from, takers or makers.

        It can be useful finding support and resistance zones.

        .. image:: images/makers_vs_takers_plot.png
           :width: 1000
           :alt: Candles with some indicators

        :param bins: How many bars.
        :param hist_funct: The way graph data is showed. It can be 'percent', 'probability', 'density', or 'probability density'
        :param height: Height of the graph.
        :param from_trades: Requieres grabbing trades before.
        :param title: A title.
        :param total_volume: The column with the total volume. It defaults automatically.
        :param partial_vol: The column with the partial volume. It defaults automatically. API shows maker or taker separated volumes.
        :param kwargs_update_layout: Optional
        :return:
        """
        if from_trades:
            if self.trades.empty:
                binpan_logger.info("Trades not downloaded. Please add trades data with: my_binpan.get_trades()")
                return
            else:
                _df = self.trades.copy(deep=True)
                if not total_volume:
                    total_volume = 'Quantity'
                if not partial_vol:
                    partial_vol = 'Buyer was maker'

                makers = _df.loc[_df[partial_vol]][total_volume]
                takers = _df.loc[~_df[partial_vol]][total_volume]

        else:
            _df = self.df.copy()
            if not total_volume:
                total_volume = 'Volume'
            if not partial_vol:
                partial_vol = 'Taker buy base volume'
            makers = _df[partial_vol]
            takers = _df[total_volume] - makers

        if not title:
            title = f"Histogram for sizes in Makers vs Takers {self.symbol} ({hist_funct})"

        handlers.plotting.plot_hists_vs(x0=makers,
                                        x1=takers,
                                        x0_name=f"{total_volume} - {partial_vol}",
                                        x1_name=partial_vol,
                                        bins=bins,
                                        hist_funct=hist_funct,
                                        height=height,
                                        title=title,
                                        **kwargs_update_layout)

    def plot_trades_scatter(self,
                            x: str = ['Price', 'Close'],
                            y: str = ['Quantity', 'Volume'],
                            dot_symbol='Buyer was maker',
                            color: str = ['Buyer was maker', 'Taker buy base volume'],
                            marginal=True,
                            from_trades=True,
                            height=1000,
                            color_referenced_to_y=True,  # useful to compare volume with taker volume for coloring
                            **kwargs):
        """
        A scatter plot showing each price level volume or trades.

        It can be useful finding support and resistance zones.

        .. image:: images/plot_trades_scatter.png
           :width: 1000
           :alt: Candles with some indicators

        :param dot_symbol: Column with discrete values to asign different symbols for the plot marks.
        :param x: Name of the column with prices. From trades or candles.
        :param y: Name of the column with sizes. From trades or candles.
        :param color: Column with values to use in color scale.
        :param marginal: Show or not lateral plots.
        :param from_trades: Uses trades instead of candles. Useful to avoid grabbing very long time intervals. Result should be similar.
        :param height: Height of the plot.
        :param color_referenced_to_y: Scales color in y axis.
        :param kwargs: Optional plotly args.
        """
        if self.trades.empty and from_trades:
            binpan_logger.info("Trades not downloaded. Please add trades data with: my_binpan.get_trades()")
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
            handlers.plotting.plot_scatter(df=data,
                                           x_col=x,
                                           y_col=y,
                                           color=color,
                                           marginal=marginal,
                                           title=title,
                                           height=height,
                                           **kwargs)
        else:
            data = self.trades.copy(deep=True)
            if not (type(x) == str and type(y) == str) and type(color):
                x = x[0]
                y = y[0]
                color = color[0]
            title = f"Priced volume for {self.symbol} data obtained from historical trades."
            handlers.plotting.plot_scatter(df=data,
                                           x_col=x,
                                           y_col=y,
                                           symbol=dot_symbol,
                                           color=color,
                                           marginal=marginal,
                                           title=title,
                                           height=height,
                                           **kwargs)

    def get_fees(self, symbol: str = None):
        """
        Shows applied fees for the symbol of the object.

        Requires API key added. Look for the add_api_key function in the files_and_filters submodule.

        :param symbol: Not to use it, just here for initializing the class.
        :return: Dictionary
        """
        try:
            if not symbol:
                symbol = self.symbol
            return handlers.wallet.get_fees(symbol=symbol)

        except NameError:
            binpan_logger.warning("Fees cannot be requested without api key added. Add it with"
                                  " binpan.handlers.files_filters.add_api_key('xxxxxxxxxx')")

    @staticmethod
    def parse_candles_to_dataframe(response: list,
                                   columns: list,
                                   symbol: str,
                                   tick_interval: str,
                                   time_cols: list,
                                   time_zone: str = None,
                                   time_index=False) -> pd.DataFrame:
        """
        Format a list of lists by changing the indicated time fields to string format.

        Passing a time_zone, for example 'Europe/Madrid', will change the time from utc to the indicated zone.

        It will automatically sort the DataFrame using the first column of the time_cols list.

        The index of the DataFrame will be numeric correlative.

        :param list(lists) response:        API klines response. List of lists.
        :param list columns:         Column names.
        :param str symbol:          Symbol requested
        :param str tick_interval:   Tick interval between candles.
        :param list time_cols:       Columns to take dates from.
        :param str time_zone:       Time zone to convert dates.
        :param bool time_index:      True gets dates index, False just numeric index.
        :return:                Pandas DataFrame

        """
        df = pd.DataFrame(response, columns=columns)

        for col in df.columns:
            df[col] = pd.to_numeric(arg=df[col], downcast='integer')

        df.loc[:, 'Open timestamp'] = df['Open time']
        df.loc[:, 'Close timestamp'] = df['Close time']

        if time_zone != 'UTC':  # converts to time zone the time columns
            for col in time_cols:
                df.loc[:, col] = handlers.time_helper.convert_utc_ms_column_to_time_zone(df, col, time_zone=time_zone)
                df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_datetime_to_string(x))
        else:
            for col in time_cols:
                df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_milliseconds_to_utc_string(x))

        if time_index:
            date_index = df['Open timestamp'].apply(handlers.time_helper.convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
            df.set_index(date_index, inplace=True)

        index_name = f"{symbol} {tick_interval} {time_zone}"
        df.index.name = index_name
        return df

    @staticmethod
    def parse_agg_trades_to_dataframe(response: list,
                                      columns: dict,
                                      symbol: str,
                                      time_zone: str = None,
                                      time_index: bool = None):
        """
        Parses the API response into a pandas dataframe.

        .. code-block::

             {'M': True,
              'T': 1656166914571,
              'a': 1218761712,
              'f': 1424997754,
              'l': 1424997754,
              'm': True,
              'p': '21185.05000000',
              'q': '0.03395000'}

        :param list response: API raw response from trades.
        :param columns: Column names.
        :param symbol: The used symbol.
        :param time_zone: Selected time zone.
        :param time_index: Or integer index.
        :return: pd.DataFrame
        """
        df = pd.DataFrame(response)
        df.rename(columns=columns, inplace=True)
        df.loc[:, 'Buyer was maker'] = df['Buyer was maker'].replace({'Maker buyer': 1, 'Taker buyer': 0})

        for col in df.columns:
            df[col] = pd.to_numeric(arg=df[col], downcast='integer')

        timestamps_serie = df['Timestamp']
        col = 'Timestamp'
        if time_zone != 'UTC':  # converts to time zone the time columns
            df.loc[:, col] = handlers.time_helper.convert_utc_ms_column_to_time_zone(df, col, time_zone=time_zone)
            df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_datetime_to_string(x))
        else:
            df.loc[:, col] = df[col].apply(lambda x: handlers.time_helper.convert_milliseconds_to_utc_string(x))

        if time_index:
            date_index = timestamps_serie.apply(handlers.time_helper.convert_milliseconds_to_time_zone_datetime, timezoned=time_zone)
            df.set_index(date_index, inplace=True)

        index_name = f"{symbol} {time_zone}"
        df.index.name = index_name
        df.loc[:, 'Buyer was maker'] = df['Buyer was maker'].astype(bool)
        return df

    @staticmethod
    def basic_dataframe(data: pd.DataFrame,
                        exceptions: list = None,
                        actions_col='actions'
                        ) -> pd.DataFrame:
        """
        Delete all columns except: Open, High, Low, Close, Volume, actions.

        Some columns can be excepted.

        Useful to drop messed technical indicators columns in one shot.

        :param pd.DataFrame data:        A BinPan DataFrame
        :param list exceptions:  A list of columns to avoid dropping.
        :param str actions_col: A column with operative actions.
        :return pd.DataFrame: Pandas DataFrame

        """
        df_ = data.copy(deep=True)
        if actions_col not in df_.columns:
            if exceptions:
                return df_[['Open', 'High', 'Low', 'Close', 'Volume'] + exceptions].copy(deep=True)
            else:
                return df_[['Open', 'High', 'Low', 'Close', 'Volume']].copy(deep=True)
        else:
            if exceptions:
                return df_[['Open', 'High', 'Low', 'Close', 'Volume', actions_col] + exceptions].copy(deep=True)
            else:
                return df_[['Open', 'High', 'Low', 'Close', 'Volume', actions_col]].copy(deep=True)

    # INDICATORS

    def ma(self,
           ma_name: str = 'ema',
           column_source: str = 'Close',
           inplace: bool = True,
           suffix: str = None,
           color: str or int = None,
           **kwargs):
        """
        Generic moving average method. Calls pandas_ta 'ma' method.

        `<https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/ma.py>`_

        :param ma_name:
        :param column_source:
        :param inplace:
        :param suffix:
        :param color:
        :param kwargs:
        :return:
        """

        if 'length' in kwargs.keys():
            if kwargs['length'] >= len(self.df):
                msg = f"BinPan Error: Ma window larger than data length."
                binpan_logger.error(msg)
                raise Exception(msg)
        if suffix:
            kwargs.update({'suffix': suffix})

        df = self.df.copy(deep=True)
        ma = ta.ma(name=ma_name, source=df[column_source], **kwargs)

        if inplace:
            # plot ready
            column_name = str(ma.name)
            self.plot_colors(indicator_column=column_name, color=color)
            self.plot_rows(indicator_column=str(column_name), row_position=1)  # overlaps are one
            self.df.loc[:, column_name] = ma

        return ma

    def sma(self, window: int = 21, column: str = 'Close', inplace=True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Generate technical indicator Simple Moving Average.

        :param window: Rolling window including the current candles when calculating the indicator.
        :param column: Column applied. Default is Close.
        :param inplace: Make it permanent in the instance or not.
        :param suffix: A decorative suffix for the name of the column created.
        :param color: Color to show when plotting. It can be any color from plotly library or a number in the list of those.
            <https://community.plotly.com/t/plotly-colours-list/11730>
        :param kwargs: Optional plotly args.
        :return: pd.DataFrame
        """
        return self.ma(ma_name='sma', column_source=column, inplace=inplace, length=window, suffix=suffix, color=color, **kwargs)

    def ema(self, window: int = 21, column: str = 'Close', inplace=True, suffix: str = '', color: str or int = None, **kwargs):
        """
        Generate technical indicator Exponential Moving Average.

        :param window: Rolling window including the current candles when calculating the indicator.
        :param column: Column applied. Default is Close.
        :param inplace: Make it permanent in the instance or not.
        :param suffix: A decorative suffix for the name of the column created.
        :param color: Color to show when plotting. It can be any color from plotly library or a number in the list of those.
            <https://community.plotly.com/t/plotly-colours-list/11730>
        :param kwargs: Optional plotly args.
        :return: pd.DataFrame
        """
        return self.ma(ma_name='ema', column_source=column, inplace=inplace, length=window, suffix=suffix, color=color, **kwargs)

    def supertrend(self, length: int = 10, multiplier: int = 3, inplace=True, suffix: str = '', colors: list = None, **kwargs):
        """
        Generate technical indicator Supertrend.

        :param length: Rolling window including the current candles when calculating the indicator.
        :param multiplier: Indicator multiplier applied.
        :param inplace: Make it permanent in the instance or not.
        :param suffix: A decorative suffix for the name of the column created.
        :param kwargs: Optional plotly args.
        :return: pd.DataFrame
        """
        if suffix:
            kwargs.update({'suffix': suffix})
        supertrend_df = ta.supertrend(high=self.df['High'],
                                      low=self.df['Low'],
                                      close=self.df['Close'],
                                      length=length,
                                      multiplier=multiplier,
                                      **kwargs)
        supertrend_df.replace(0, np.nan, inplace=True)  # pandas_ta puts a zero at the beginning sometimes

        if inplace:
            # plot ready
            column_names = supertrend_df.columns
            self.row_counter += 1
            if not colors:
                colors = ['yellow', 'blue', 'green', 'red']
            for i, col in enumerate(column_names):
                self.plot_colors(indicator_column=col, color=colors[i])
                if col.startswith("SUPERTs_"):
                    self.plot_rows(indicator_column=col, row_position=1)  # overlaps are one
                elif col.startswith("SUPERTl_"):
                    self.plot_rows(indicator_column=col, row_position=1)  # overlaps are one
                elif col.startswith("SUPERT_"):
                    self.plot_rows(indicator_column=col, row_position=1)  # overlaps are one
                elif col.startswith("SUPERTd_"):
                    self.plot_rows(indicator_column=col, row_position=self.row_counter)  # overlaps are one
            self.df = pd.concat([self.df, supertrend_df], axis=1)
        return supertrend_df
