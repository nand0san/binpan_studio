"""

This is the main classes file.

"""

import pandas as pd
import numpy as np
import handlers.logs
import handlers.market
import handlers.quest
import handlers.exceptions
import handlers.time_helper
import handlers.plotting
import handlers.wallet
import handlers.files_filters
import handlers.strategies

import pandas_ta as ta
from random import choice

binpan_logger = handlers.logs.Logs(filename='./logs/binpan.log', name='binpan', info_level='DEBUG')
tick_seconds = handlers.time_helper.tick_seconds

__version__ = "0.0.10"

plotly_colors = handlers.plotting.plotly_colors


class Symbol(object):
    """
    Creates an object from binance klines and/or trade data. It contains the raw api response and a dataframe that can be modified.

    Any symbol can be used as argument and any time interval like: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h',
    '12h', '1d', '3d', '1w', '1M'

    Object has several plot methods.

    :param str symbol:  It can be any symbol in the binance exchange, like BTCUSDT, ethbusd or any other. Capital letters doesn't matter.

    :param str tick_interval: Any candle's interval available in binance. Capital letters doesn't matter.

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

            btcusdt = binpan.Symbol(symbol='btcusdt',
                                    tick_interval='15m',
                                    time_zone='Europe/Madrid',
                                    start_time='2021-10-31 01:00:00',
                                    end_time='2021-10-31 03:00:00')

            btcusdt.df

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

        binpan_object = binpan.Symbol(symbol='ethbusd',
                                      tick_interval='5m',
                                      limit=100,
                                      display_rows=10)
        binpan_object.df

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
        self.row_control = dict()
        self.color_control = []
        self.color_fill_control = dict()
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


        Example:

            .. code-block::

                Aggregate tradeId	Price	Quantity	First tradeId	Last tradeId	Timestamp	Buyer was maker	Best price match

                LUNCBUSD Europe/Madrid

                2022-07-03 04:51:45.633000+02:00	12076196	0.000125	9.314130e+04	17097916	17097916	2022-07-03 04:51:45	False	True
                2022-07-03 04:51:45.811000+02:00	12076197	0.000125	7.510097e+04	17097917	17097917	2022-07-03 04:51:45	False	True
                2022-07-03 04:51:45.811000+02:00	12076198	0.000125	2.204286e+06	17097918	17097918	2022-07-03 04:51:45	False	True
                2022-07-03 04:51:46.079000+02:00	12076199	0.000125	8.826475e+04	17097919	17097919	2022-07-03 04:51:46	True	True
                2022-07-03 04:51:46.811000+02:00	12076200	0.000125	5.362675e+06	17097920	17097920	2022-07-03 04:51:46	True	True
                ...	...	...	...	...	...	...	...	...
                2022-07-03 13:11:49.507000+02:00	12141092	0.000125	1.363961e+07	17183830	17183831	2022-07-03 13:11:49	False	True
                2022-07-03 13:11:49.507000+02:00	12141093	0.000125	2.000000e+06	17183832	17183832	2022-07-03 13:11:49	False	True
                2022-07-03 13:11:49.507000+02:00	12141094	0.000125	1.500000e+05	17183833	17183833	2022-07-03 13:11:49	False	True
                2022-07-03 13:11:49.507000+02:00	12141095	0.000125	1.562695e+08	17183834	17183835	2022-07-03 13:11:49	False	True
                2022-07-03 13:11:50.172000+02:00	12141096	0.000125	1.048632e+06	17183836	17183836	2022-07-03 13:11:50	False	True


        :return: Pandas DataFrame

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

    ################
    # Plots
    ################

    def set_plot_row(self, indicator_column: str = None, row_position: int = None):
        """
        Internal control formatting plots. Can be used to change plot subplot row of an indicator.

        :param str indicator_column: column name
        :param row_position: reassign row_position to column name
        :return dict: columns with its assigned row_position in subplots when plotting.

        """
        if indicator_column and row_position:
            self.row_control.update({indicator_column: row_position})
        return self.row_control

    def set_plot_color(self, indicator_column: str = None, color: int or str = None) -> list:
        """
        Internal control formatting plots. Can be used to change plot color of an indicator.

        :param str indicator_column: column name
        :param color: reassign color to column name
        :return dict: columns with its assigned colors when plotting.

        """
        binpan_logger.debug(f"set_plot_color: indicator_column:{indicator_column} color:{color}")

        existing_colors = self.color_control
        updated_colors = existing_colors
        if indicator_column and color:
            if type(color) == int:
                updated_colors = self.update_tuples_list(tuples_list=existing_colors,
                                                         k=indicator_column,
                                                         v=plotly_colors[color])
            elif color in plotly_colors:
                updated_colors = self.update_tuples_list(tuples_list=existing_colors,
                                                         k=indicator_column,
                                                         v=color)
        elif indicator_column:
            updated_colors = self.update_tuples_list(tuples_list=existing_colors,
                                                     k=indicator_column,
                                                     v=choice(plotly_colors))
        self.color_control = updated_colors
        binpan_logger.debug(f"Updated self.color_control: {updated_colors}")
        return self.color_control

    def set_plot_color_fill(self, indicator_column: str = None, color_fill: str or bool = None) -> dict:
        """
        Internal control formatting plots. Can be used to change plot color of an indicator.

        :param str indicator_column: column name
        :param color_fill: Color can be forced to fill to zero line. For transparent colors use rgba string code to define color.
         Example for transparent green 'rgba(26,150,65,0.5)' or transparent red 'rgba(204,0,0,0.5)'
        :return dict: columns with its assigned colors when plotting.

        """
        if indicator_column and color_fill:
            if type(color_fill) == int:
                self.color_fill_control.update({indicator_column: plotly_colors[color_fill]})
            elif color_fill in plotly_colors or color_fill.startswith('rgba'):
                print("fill detected")
                self.color_fill_control.update({indicator_column: color_fill})
        elif indicator_column:
            self.color_fill_control.update({indicator_column: None})
        return self.color_fill_control

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
        :param candles_ta_height_ratio: Proportion between candles and the other indicators. Not considering overlap ones
         in the candles plot.
        :param plot_volume: Plots volume.
        :param title: A tittle for the plot.
        :param yaxis_title: A title for the y axis.
        :param overlapped_indicators: Can declare as overlap in the candles plot some column.
        :param priced_actions_col: Priced actions to plot annotations over the candles, like buy, sell, etc. Under developing.
        :param actions_col: A column containing actions like buy or sell. Under developing.
        :param labels: Names for the annotations instead of the price.
        :param default_price_for_actions: Column to use as priced actions in case of not existing a specific prices actions column.
        """
        binpan_logger.debug(f"PLOT: self.row_control:{self.row_control} self.color_control:{self.color_control} ")
        if not title:
            title = self.df.index.name

        indicators_series = [self.df[k] for k in self.row_control.keys()]
        indicator_names = [self.df[k].name for k in self.row_control.keys()]
        # indicators_colors = [self.color_control[k] for k in self.row_control.keys()]
        indicators_colors = [self.color_control[i][1] for i, k in enumerate(self.row_control)]
        binpan_logger.debug(f"PLOT: indicator_names: {indicator_names} indicators_colors:{indicators_colors}")
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
                                         fill_control=self.color_fill_control,
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
        :param logarithmic: Logarithmic scale to show more small sizes.
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

    def plot_makers_vs_takers(self,
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

        Can be useful finding support and resistance zones.

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

    #################
    # Exchange Data #
    #################

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

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def update_tuples_list(tuples_list: list, k, v):
        result = []
        updated = False
        for i in tuples_list:
            key = i[0]
            value = i[1]
            if k == key:
                ret = (k, v)
                updated = True
            else:
                ret = (key, value)
            result.append(ret)
        if not updated:
            result.append((k, v))
        return result

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
        :return: Pandas DataFrame

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

    ##############
    # Indicators #
    ##############

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

        :param str ma_name: A moving average supported by the generic pandas_ta "ma" function.
        :param str column_source: Name of column with data to be used.
        :param bool inplace: Permanent or not.
        :param str suffix: A string to decorate resulting pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :param kwargs: From https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/ma.py
        :return: pd.Series

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
        """
        return self.ma(ma_name='ema', column_source=column, inplace=inplace, length=window, suffix=suffix, color=color, **kwargs)

    def supertrend(self, length: int = 10, multiplier: int = 3, inplace=True, suffix: str = None, colors: list = None, **kwargs):
        """
        Generate technical indicator Supertrend.

        :param int length: Rolling window including the current candles when calculating the indicator.
        :param int multiplier: Indicator multiplier applied.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: DDefaults to red and green.
        :param kwargs: Optional plotly args from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/supertrend.py.
        :return: pd.DataFrame

        """
        if suffix:
            kwargs.update({'suffix': suffix})
        supertrend_df = ta.supertrend(high=self.df['High'],
                                      low=self.df['Low'],
                                      close=self.df['Close'],
                                      length=length,
                                      multiplier=int(multiplier),
                                      **kwargs)
        supertrend_df.replace(0, np.nan, inplace=True)  # pandas_ta puts a zero at the beginning sometimes that can break the plot scale

        if inplace:
            column_names = supertrend_df.columns
            self.row_counter += 1
            if not colors:
                colors = ['yellow', 'blue', 'green', 'red']
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

    def macd(self, fast: int = 12,
             slow: int = 26,
             smooth: int = 9,
             inplace: bool = True,
             suffix: str = '',
             colors: list = ['orange', 'green', 'skyblue'],
             **kwargs):
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

        """
        macd = self.df.ta.macd(fast=fast,
                               slow=slow,
                               signal=smooth,
                               **kwargs)
        if inplace:
            self.row_counter += 1
            for i, c in enumerate(macd.columns):
                col = macd[c]
                column_name = str(col.name) + suffix
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                if c.startswith('MACDh_'):
                    self.set_plot_color_fill(indicator_column=column_name, color_fill='rgba(26,150,65,0.5)')
                else:
                    self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
                self.df.loc[:, column_name] = col
        return macd

    def rsi(self,
            length: int = 14,
            inplace: bool = True,
            suffix: str = '',
            color: str or int = None,
            **kwargs):
        """
        Relative Strength Index (RSI).

            https://www.investopedia.com/terms/r/rsi.asp

        :param int length: Default is 21
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting from plotly list or index of color in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/rsi.py
        :return: A Pandas Series
        """

        rsi = ta.rsi(close=self.df['Close'],
                     length=length,
                     **kwargs)
        column_name = str(rsi.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'orange'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = rsi
        return rsi

    def stoch_rsi(self,
                  rsi_length: int = 14,
                  k_smooth: int = 3,
                  d_smooth: int = 3,
                  inplace: bool = True,
                  suffix: str = '',
                  colors: list = ['orange', 'blue'],
                  **kwargs) -> pd.DataFrame:
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
        """


        binpan_logger.debug(f"COLOR control inicial: {self.color_control}")

        stoch_df = ta.stochrsi(close=self.df['Close'],
                               length=rsi_length,
                               rsi_length=rsi_length,
                               k_smooth=k_smooth,
                               d_smooth=d_smooth,
                               **kwargs)

        binpan_logger.debug(f"len:{len(stoch_df)} type:{type(stoch_df)}")
        if inplace:
            self.row_counter += 1
            for i, c in enumerate(stoch_df.columns):
                serie = stoch_df[c]
                column_name = str(serie.name) + suffix
                binpan_logger.debug(f"Vamos a llamar a set_plot_color con: {column_name} {colors[i]}")
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
                self.df.loc[:, column_name] = serie
                binpan_logger.debug(f"resultados actualizados: {column_name} {colors[i]} self.color_control: {self.color_control}")

        self.set_plot_color(indicator_column='STOCHRSId_14_14_3_3', color='green')
        self.set_plot_color(indicator_column='STOCHRSId_14_14_3_3', color='bluesky')
        self.set_plot_color(indicator_column='STOCHRSId_14_14_3_3', color='bluesky')

        binpan_logger.debug(f"resultados actualizados fin: self.color_control: {self.color_control}")
        return stoch_df

    def on_balance_volume(self,
                          inplace: bool = True,
                          suffix: str = '',
                          color: str or int = None,
                          **kwargs):
        """
        On balance indicator.

            https://www.investopedia.com/terms/o/onbalancevolume.asp

        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volume/obv.py
        :return: A Pandas Series
        """

        on_balance = ta.obv(close=self.df['Close'],
                            volume=self.df['Volume'],
                            **kwargs)

        column_name = str(on_balance.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'red'

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = on_balance

        return on_balance

    def accumulation_distribution(self,
                                  inplace: bool = True,
                                  suffix: str = '',
                                  color: str or int = None,
                                  **kwargs):
        """
        Accumulation/Distribution indicator.

            https://www.investopedia.com/terms/a/accumulationdistribution.asp

        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volume/ad.py
        :return: A Pandas Series
        """

        ad = ta.ad(high=self.df['High'],
                   low=self.df['Low'],
                   close=self.df['Close'],
                   volume=self.df['Volume'],
                   **kwargs)

        column_name = str(ad.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'red'

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = ad

        return ad

    def vwap(self,
             anchor: str = "D",
             inplace: bool = True,
             suffix: str = '',
             color: str or int = None,
             **kwargs):
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
        """

        vwap = ta.vwap(high=self.df['High'],
                       low=self.df['Low'],
                       close=self.df['Close'],
                       volume=self.df['Volume'],
                       anchor=anchor,
                       **kwargs)

        column_name = str(vwap.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = vwap

        return vwap

    def atr(self,
            length: int = 14,
            inplace: bool = True,
            suffix: str = '',
            color: str or int = None,
            **kwargs):
        """
        Average True Range.

            https://www.investopedia.com/terms/a/atr.asp

        :param str length: Window period to obtain ATR. Default is 14.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volatility/atr.py
        :return: A Pandas Series
        """

        atr = ta.atr(high=self.df['High'],
                     low=self.df['Low'],
                     close=self.df['Close'],
                     length=length,
                     **kwargs)

        column_name = str(atr.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = atr

        return atr

    def cci(self,
            length: int = 14,
            scaling: int = None,
            inplace: bool = True,
            suffix: str = '',
            color: str or int = None,
            **kwargs):
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
        """

        cci = ta.cci(high=self.df['High'],
                     low=self.df['Low'],
                     close=self.df['Close'],
                     length=length,
                     c=scaling,
                     **kwargs)

        column_name = str(cci.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = cci

        return cci

    def eom(self,
            length: int = 14,
            divisor: int = None,
            drift: int = None,
            inplace: bool = True,
            suffix: str = '',
            color: str or int = None,
            **kwargs):
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
        """

        eom = ta.eom(high=self.df['High'],
                     low=self.df['Low'],
                     close=self.df['Close'],
                     volume=self.df['Volume'],
                     length=length,
                     divisor=divisor,
                     drift=drift,
                     **kwargs)

        column_name = str(eom.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = eom
        return eom

    def roc(self,
            length: int = 1,
            escalar: int = 100,
            inplace: bool = True,
            suffix: str = '',
            color: str or int = None,
            **kwargs):
        """
        The Rate of Change (ROC) is a technical indicator that measures the percentage change between the most recent price
            and the price "n" day’s ago. The indicator fluctuates around the zero line.

                https://blog.quantinsti.com/build-technical-indicators-in-python/

        :param str length: The short period. Default: 1
        :param str escalar:  How much to magnify. Default: 100.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param str or int color: Is the color to show when plotting or index in that list.
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/roc.py
        :return: A Pandas Series
        """

        roc = ta.roc(close=self.df['Close'],
                     length=length,
                     escalar=escalar,
                     **kwargs)

        column_name = str(roc.name) + suffix

        if inplace:
            self.row_counter += 1
            if not color:
                color = 'red'
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
            self.df.loc[:, column_name] = roc
        return roc

    def bbands(self,
               length: int = 5,
               std: int = 2,
               ddof: int = 0,
               inplace: bool = True,
               suffix: str = '',
               colors: list = ['green', 'orange', 'red', 'skyblue'],
               **kwargs):
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

        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volatility/bbands.py
        :return: pd.Series

        """
        bbands = self.df.ta.bbands(close=self.df['Close'],
                                   length=length,
                                   std=std,
                                   ddof=ddof,
                                   suffix=suffix,
                                   **kwargs)
        if inplace:
            self.row_counter += 1
            for i, c in enumerate(bbands.columns):
                col = bbands[c]
                column_name = str(col.name)
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                # TODO: definir fill color por nombre de columna
                if c.startswith('MACDh_'):
                    self.set_plot_color_fill(indicator_column=column_name, color_fill='rgba(26,150,65,0.5)')
                else:
                    self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)
                self.df.loc[:, column_name] = col
        return bbands

    @staticmethod
    def pandas_ta_indicator(name: str,
                            **kwargs):
        """
        Calls any indicator in pandas_ta library with function name as first argument and any kwargs the function will use.

        :param name: A function name. In example: 'massi' for Mass Index or 'rsi' for RSI indicator.
        :param kwargs: Arguments for the requested indicator. Review pandas_ta info: https://github.com/twopirllc/pandas-ta#features
        :return: Whatever returns pandas_ta
        """

        if name == "ebsw":
            return ta.ebsw(**kwargs)
        elif name == "ao":
            return ta.ao(**kwargs)
        elif name == "apo":
            return ta.apo(**kwargs)
        elif name == "bias":
            return ta.bias(**kwargs)
        elif name == "bop":
            return ta.bop(**kwargs)
        elif name == "brar":
            return ta.brar(**kwargs)
        elif name == "cci":
            return ta.cci(**kwargs)
        elif name == "cfo":
            return ta.cfo(**kwargs)
        elif name == "cg":
            return ta.cg(**kwargs)
        elif name == "cmo":
            return ta.cmo(**kwargs)
        elif name == "coppock":
            return ta.coppock(**kwargs)
        elif name == "cti":
            return ta.cti(**kwargs)
        elif name == "dm":
            return ta.dm(**kwargs)
        elif name == "er":
            return ta.er(**kwargs)
        elif name == "eri":
            return ta.eri(**kwargs)
        elif name == "fisher":
            return ta.fisher(**kwargs)
        elif name == "inertia":
            return ta.inertia(**kwargs)
        elif name == "kdj":
            return ta.kdj(**kwargs)
        elif name == "kst":
            return ta.kst(**kwargs)
        elif name == "macd":
            return ta.macd(**kwargs)
        elif name == "mom":
            return ta.mom(**kwargs)
        elif name == "pgo":
            return ta.pgo(**kwargs)
        elif name == "ppo":
            return ta.ppo(**kwargs)
        elif name == "psl":
            return ta.psl(**kwargs)
        elif name == "pvo":
            return ta.pvo(**kwargs)
        elif name == "qqe":
            return ta.qqe(**kwargs)
        elif name == "roc":
            return ta.roc(**kwargs)
        elif name == "rsi":
            return ta.rsi(**kwargs)
        elif name == "rsx":
            return ta.rsx(**kwargs)
        elif name == "rvgi":
            return ta.rvgi(**kwargs)
        elif name == "stc":
            return ta.stc(**kwargs)
        elif name == "slope":
            return ta.slope(**kwargs)
        elif name == "squeeze":
            return ta.squeeze(**kwargs)
        elif name == "squeeze_pro":
            return ta.squeeze_pro(**kwargs)
        elif name == "stoch":
            return ta.stoch(**kwargs)
        elif name == "stochrsi":
            return ta.stochrsi(**kwargs)
        elif name == "td_seq":
            return ta.td_seq(**kwargs)
        elif name == "trix":
            return ta.trix(**kwargs)
        elif name == "tsi":
            return ta.tsi(**kwargs)
        elif name == "uo":
            return ta.uo(**kwargs)
        elif name == "willr":
            return ta.willr(**kwargs)
        elif name == "alma":
            return ta.alma(**kwargs)
        elif name == "dema":
            return ta.dema(**kwargs)
        elif name == "ema":
            return ta.ema(**kwargs)
        elif name == "fwma":
            return ta.fwma(**kwargs)
        elif name == "hilo":
            return ta.hilo(**kwargs)
        elif name == "hl2":
            return ta.hl2(**kwargs)
        elif name == "hlc3":
            return ta.hlc3(**kwargs)
        elif name == "hma":
            return ta.hma(**kwargs)
        elif name == "hwma":
            return ta.hwma(**kwargs)
        elif name == "ichimoku":
            return ta.ichimoku(**kwargs)
        elif name == "jma":
            return ta.jma(**kwargs)
        elif name == "kama":
            return ta.kama(**kwargs)
        elif name == "linreg":
            return ta.linreg(**kwargs)
        elif name == "mcgd":
            return ta.mcgd(**kwargs)
        elif name == "midpoint":
            return ta.midpoint(**kwargs)
        elif name == "midprice":
            return ta.midprice(**kwargs)
        elif name == "ohlc4":
            return ta.ohlc4(**kwargs)
        elif name == "pwma":
            return ta.pwma(**kwargs)
        elif name == "rma":
            return ta.rma(**kwargs)
        elif name == "sinwma":
            return ta.sinwma(**kwargs)
        elif name == "sma":
            return ta.sma(**kwargs)
        elif name == "ssf":
            return ta.ssf(**kwargs)
        elif name == "supertrend":
            return ta.supertrend(**kwargs)
        elif name == "swma":
            return ta.swma(**kwargs)
        elif name == "t3":
            return ta.t3(**kwargs)
        elif name == "tema":
            return ta.tema(**kwargs)
        elif name == "trima":
            return ta.trima(**kwargs)
        elif name == "vidya":
            return ta.vidya(**kwargs)
        elif name == "vwap":
            return ta.vwap(**kwargs)
        elif name == "vwma":
            return ta.vwma(**kwargs)
        elif name == "wcp":
            return ta.wcp(**kwargs)
        elif name == "wma":
            return ta.wma(**kwargs)
        elif name == "zlma":
            return ta.zlma(**kwargs)
        elif name == "drawdown":
            return ta.drawdown(**kwargs)
        elif name == "log_return":
            return ta.log_return(**kwargs)
        elif name == "percent_return":
            return ta.percent_return(**kwargs)
        elif name == "entropy":
            return ta.entropy(**kwargs)
        elif name == "kurtosis":
            return ta.kurtosis(**kwargs)
        elif name == "mad":
            return ta.mad(**kwargs)
        elif name == "median":
            return ta.median(**kwargs)
        elif name == "quantile":
            return ta.quantile(**kwargs)
        elif name == "skew":
            return ta.skew(**kwargs)
        elif name == "stdev":
            return ta.stdev(**kwargs)
        elif name == "tos_stdevall":
            return ta.tos_stdevall(**kwargs)
        elif name == "variance":
            return ta.variance(**kwargs)
        elif name == "zscore":
            return ta.zscore(**kwargs)
        elif name == "adx":
            return ta.adx(**kwargs)
        elif name == "amat":
            return ta.amat(**kwargs)
        elif name == "aroon":
            return ta.aroon(**kwargs)
        elif name == "chop":
            return ta.chop(**kwargs)
        elif name == "cksp":
            return ta.cksp(**kwargs)
        elif name == "decay":
            return ta.decay(**kwargs)
        elif name == "decreasing":
            return ta.decreasing(**kwargs)
        elif name == "dpo":
            return ta.dpo(**kwargs)
        elif name == "increasing":
            return ta.increasing(**kwargs)
        elif name == "long_run":
            return ta.long_run(**kwargs)
        elif name == "psar":
            return ta.psar(**kwargs)
        elif name == "qstick":
            return ta.qstick(**kwargs)
        elif name == "short_run":
            return ta.short_run(**kwargs)
        elif name == "tsignals":
            return ta.tsignals(**kwargs)
        elif name == "ttm_trend":
            return ta.ttm_trend(**kwargs)
        elif name == "vhf":
            return ta.vhf(**kwargs)
        elif name == "vortex":
            return ta.vortex(**kwargs)
        elif name == "xsignals":
            return ta.xsignals(**kwargs)
        elif name == "above":
            return ta.above(**kwargs)
        elif name == "above_value":
            return ta.above_value(**kwargs)
        elif name == "below":
            return ta.below(**kwargs)
        elif name == "below_value":
            return ta.below_value(**kwargs)
        elif name == "cross":
            return ta.cross(**kwargs)
        elif name == "aberration":
            return ta.aberration(**kwargs)
        elif name == "accbands":
            return ta.accbands(**kwargs)
        elif name == "atr":
            return ta.atr(**kwargs)
        elif name == "bbands":
            return ta.bbands(**kwargs)
        elif name == "donchian":
            return ta.donchian(**kwargs)
        elif name == "hwc":
            return ta.hwc(**kwargs)
        elif name == "kc":
            return ta.kc(**kwargs)
        elif name == "massi":
            return ta.massi(**kwargs)
        elif name == "natr":
            return ta.natr(**kwargs)
        elif name == "pdist":
            return ta.pdist(**kwargs)
        elif name == "rvi":
            return ta.rvi(**kwargs)
        elif name == "thermo":
            return ta.thermo(**kwargs)
        elif name == "true_range":
            return ta.true_range(**kwargs)
        elif name == "ui":
            return ta.ui(**kwargs)
        elif name == "ad":
            return ta.ad(**kwargs)
        elif name == "adosc":
            return ta.adosc(**kwargs)
        elif name == "aobv":
            return ta.aobv(**kwargs)
        elif name == "cmf":
            return ta.cmf(**kwargs)
        elif name == "efi":
            return ta.efi(**kwargs)
        elif name == "eom":
            return ta.eom(**kwargs)
        elif name == "kvo":
            return ta.kvo(**kwargs)
        elif name == "mfi":
            return ta.mfi(**kwargs)
        elif name == "nvi":
            return ta.nvi(**kwargs)
        elif name == "obv":
            return ta.obv(**kwargs)
        elif name == "pvi":
            return ta.pvi(**kwargs)
        elif name == "pvol":
            return ta.pvol(**kwargs)
        elif name == "pvr":
            return ta.pvr(**kwargs)
        elif name == "pvt":
            return ta.pvt(**kwargs)
        elif name == "vp":
            return ta.vp(**kwargs)
