"""

This is the main classes file.

"""
__version__ = "0.8.10"

import os
from sys import path
import pandas as pd
from typing import Tuple, List, Union, Optional
import pandas_ta as ta
from random import choice
from time import time
import numpy as np

from binpan.exchange_manager import Exchange
from .auxiliar import csv_klines_setup, check_continuity, setup_startime_endtime, repair_kline_discontinuity

from handlers.exceptions import BinPanException
from handlers.exchange import (get_decimal_positions, get_info_dic, get_precision, get_orderTypes_and_permissions, get_fees,
                               get_symbols_filters)
from handlers.files import select_file, read_csv_to_dataframe, save_dataframe_to_csv, extract_filename_metadata, get_encoded_secrets

from handlers.indicators import (df_splitter, reversal_candles, zoom_cloud_indicators, ichimoku, fractal_w_indicator,
                                 support_resistance_levels, ffill_indicator, shift_indicator, market_profile_from_klines_melt,
                                 alternating_fractal_indicator, fractal_trend_indicator, market_profile_from_klines_grouped,
                                 market_profile_from_trades_grouped, support_resistance_levels_merged, time_active_zones)

from handlers.logs import Logs

from handlers.market import (get_candles_by_time_stamps, parse_candles_to_dataframe, convert_to_numeric, basic_dataframe,
                             get_historical_agg_trades, parse_agg_trades_to_dataframe, get_historical_atomic_trades,
                             parse_atomic_trades_to_dataframe, get_order_book)

from handlers.plotting import (plotly_colors, plot_trades, candles_ta, plot_pie, plot_hists_vs, candles_tagged, bar_plot, plot_scatter,
                               orderbook_depth, dist_plot, profile_plot)

from handlers.time_helper import (check_tick_interval, convert_milliseconds_to_str, get_dataframe_time_index_ranges,
                                  remove_initial_included_ranges, tick_interval_values, pandas_freq_tick_interval)

from handlers.tags import (tag_column_to_strategy_group, backtesting, backtesting_short, tag_comparison, tag_cross, merge_series,
                           clean_in_out)

from handlers.wallet import convert_str_date_to_ms

from handlers.aggregations import resample_klines

from handlers.standards import (binance_api_candles_cols, agg_trades_columns, atomic_trades_columns, time_cols,
                                dts_time_cols, reversal_columns, agg_trades_columns_from_binance, atomic_trades_columns_from_binance)

from handlers.quest import tick_seconds

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from handlers.numba_tools import sma_numba, rsi_numba, ema_numba
# try:
#     from numba import njit
#     from handlers.numba_tools import sma_numba, rsi_numba, ema_numba
#     is_numba = True
# except ImportError:
#     is_numba = False
#     sma_numba, rsi_numba, ema_numba = None, None, None

binpan_logger = Logs(filename='./logs/binpan.log', name='binpan', info_level='INFO')
version = __version__

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 250)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 12)

empty_agg_trades_msg = "Empty trades, please request using: get_agg_trades() method: Example: my_symbol.get_agg_trades()"
empty_atomic_trades_msg = "Empty atomic trades, please request using: get_atomic_trades() method: Example: my_symbol.get_atomic_trades()"


class Symbol(object):
    """
    Creates an object from binance klines and/or trade data. It contains the raw api response, a dataframe that can be modified and many
    more.

    Symbols can be any trading pair on Binance, such as BTCUSDT or ETHBUSD, and time intervals can be specified using the
    following strings:

        ```
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', or '1M'
        ```

    The class provides several plotting methods for quick data visualization.

    :param str symbol:  It can be any symbol in the binance exchange, like BTCUSDT, ethbusd or any other. Capital letters don't matter.
    :param str tick_interval: Any candle's interval available in binance. Capital letters don't matter.
    :param int or str start_time:  It can be an integer in milliseconds from epoch (1970-01-01 00:00:00 UTC) or any string in the formats:

      .. code-block::

        - %Y-%m-%d %H:%M:%S.%f:       **2022-05-11 06:45:42.124567**
        - %Y-%m-%d %H:%M:%S:          **2022-05-11 06:45:42**

     If start time is passed, it gets the next open according to the tick interval selected except an exact open time passed.

    :param int or str end_time:    It can be an integer in milliseconds from epoch (1970-01-01 00:00:00 UTC) or any string in the formats:

      .. code-block::

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

    :param bool closed:      The last candle is a closed one in the moment of the creation, discarding the current running one not closed yet. Default True.
    :param int display_columns:     Number of columns in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param int display_max_rows:        Number of rows in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param int display_width:       Display width in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param int display_min_rows:        Number of rows in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param bool or str postgres_klines:    If True, gets data from a postgres database by selecting interactively from tables found.
        Also a string with table name can be used. If str passed, it will be used as host.
    :param bool postgres_agg_trades:    If True, gets data from a postgres database by selecting interactively from tables found.
        Also a string with table name can be used. If str passed, it will be used as host.
    :param bool postgres_atomic_trades:    If True, gets data from a postgres database by selecting interactively from tables found.
        Also a string with table name can be used. If str passed, it will be used as host.
    :param int hours:    If hours is passed, it gets the candles from the last hours.
    :param bool from_csv:    If True, gets data from a csv file by selecting interactively from csv files found.
        Also a string with filename can be used.
    :param dict info_dic:               Sometimes, for iterative processes, info_dic can be passed to avoid calling API for it. Its
        weight is heavy.
    :param bool or str from_csv:    If True, gets data from a csv file by selecting interactively from csv files found.
     Also a string with filename can be used.
    :param dict info_dic:               Sometimes, for iterative processes, info_dic can be passed to avoid calling API for it. Its
     weight is heavy.

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

    """

    def __init__(self,
                 symbol: str = None,
                 tick_interval: str = None,
                 start_time: int or str = None,
                 end_time: int or str = None,
                 limit: int = 1000,
                 time_zone: str = 'Europe/Madrid',
                 closed: bool = True,
                 hours: int = None,
                 postgres_klines: bool or str = False,
                 postgres_agg_trades: bool or str = False,
                 postgres_atomic_trades: bool or str = False,
                 display_columns: int = 25,
                 display_max_rows: int = 10,
                 display_min_rows: int = 25,
                 display_width: int = 320,
                 from_csv: Union[bool, str] = False,
                 info_dic: dict = None):

        if not symbol and not from_csv:
            raise BinPanException(f"BinPan Exception: symbol needed")

        if not from_csv and not symbol.isalnum():
            binpan_logger.error(f"BinPan Exception: Ilegal characters in symbol.")

        # check correct tick interval passed
        if not from_csv:
            tick_interval = check_tick_interval(tick_interval)
        else:
            tick_interval = tick_interval

        # self.is_numba = is_numba
        self.tick_interval = tick_interval
        self.time_zone = time_zone

        # dataframe columns
        # self.presentation_columns = presentation_columns
        self.original_columns = binance_api_candles_cols

        # # trades columns provisional
        self.agg_trades_columns = agg_trades_columns
        self.atomic_trades_columns = atomic_trades_columns

        self.reversal_columns = reversal_columns

        # time cols
        self.time_cols = time_cols
        self.dts_time_cols = dts_time_cols

        # version
        self.version = __version__

        # set de rutas
        self.cwd = os.getcwd()
        path.append(self.cwd)

        # parámetros principales
        if symbol:  # en csv no se pasa symbol y da error el upper
            self.symbol = symbol.upper()
        else:
            self.symbol = ""

        if from_csv:
            (self.df,
             self.symbol,
             self.tick_interval,
             self.time_zone,
             self.data_type,
             self.start_time,
             self.end_time,
             self.closed,
             self.limit) = csv_klines_setup(from_csv=from_csv,
                                            symbol=self.symbol,
                                            tick_interval=self.tick_interval,
                                            cwd=self.cwd,
                                            time_zone=self.time_zone)
        else:
            self.symbol = symbol.upper()
            self.tick_interval = tick_interval
            self.time_zone = time_zone
            self.limit = limit
            self.closed = closed

        # database
        if postgres_klines or postgres_agg_trades or postgres_atomic_trades:

            import handlers.postgresql as postgresql
            from secret import (postgresql_port, postgresql_user, postgresql_database)

            if postgres_klines:
                if type(postgres_klines) == str:
                    binpan_logger.info(f"Postgres connection requested as str: {postgres_klines}")
                    postgresql_host_klines = postgres_klines
                else:
                    from secret import postgresql_host_klines
                self.connection_klines, self.cursor_klines = postgresql.setup(symbol=self.symbol,
                                                                              tick_interval=self.tick_interval,
                                                                              postgresql_host=postgresql_host_klines,
                                                                              postgresql_user=postgresql_user,
                                                                              postgresql_database=postgresql_database,
                                                                              postgres_klines=True,
                                                                              postgres_agg_trades=False,
                                                                              postgres_atomic_trades=False,
                                                                              postgresql_port=postgresql_port)
            if postgres_agg_trades:
                if type(postgres_agg_trades) == str:
                    postgresql_host_aggTrades = postgres_agg_trades
                else:
                    from secret import postgresql_host_aggTrades
                self.connection_agg_trades, self.cursor_agg_trades = postgresql.setup(symbol=self.symbol,
                                                                                      tick_interval=self.tick_interval,
                                                                                      postgresql_host=postgresql_host_aggTrades,
                                                                                      postgresql_user=postgresql_user,
                                                                                      postgresql_database=postgresql_database,
                                                                                      postgres_klines=False,
                                                                                      postgres_agg_trades=True,
                                                                                      postgres_atomic_trades=False,
                                                                                      postgresql_port=postgresql_port)
            if postgres_atomic_trades:
                if type(postgres_atomic_trades) == str:
                    postgresql_host_trades = postgres_atomic_trades
                else:
                    from secret import postgresql_host_trades
                self.connection_atomic_trades, self.cursor_atomic_trades = postgresql.setup(symbol=self.symbol,
                                                                                            tick_interval=self.tick_interval,
                                                                                            postgresql_host=postgresql_host_trades,
                                                                                            postgresql_user=postgresql_user,
                                                                                            postgresql_database=postgresql_database,
                                                                                            postgres_klines=False,
                                                                                            postgres_agg_trades=False,
                                                                                            postgres_atomic_trades=True,
                                                                                            postgresql_port=postgresql_port)
        else:
            self.connection_klines, self.cursor_klines = None, None

        self.postgres_klines = postgres_klines
        self.postgres_agg_trades = postgres_agg_trades
        self.postgres_atomic_trades = postgres_atomic_trades

        # pandas visualization settings
        self.display_columns = display_columns
        self.display_max_rows = display_max_rows
        self.display_min_rows = display_min_rows
        self.display_width = display_width
        self.set_display_columns(display_columns)
        self.set_display_width(display_width)
        self.set_display_min_rows(display_min_rows)
        self.set_display_max_rows(display_max_rows)

        # indicators and relevant data initialization #

        self.support_lines = None  # support levels from trades
        self.resistance_lines = None  # support levels from trades

        self.raw_agg_trades = []
        self.agg_trades = pd.DataFrame(columns=list(self.agg_trades_columns.values()))

        self.raw_atomic_trades = []
        self.atomic_trades = pd.DataFrame(columns=list(self.atomic_trades_columns.values()))

        self.min_height = 7
        self.min_reversal = 4
        self.reversal_agg_klines = pd.DataFrame(columns=self.reversal_columns)
        self.reversal_atomic_klines = pd.DataFrame(columns=self.reversal_columns)

        self.orderbook_value = None
        self.orderbook = pd.DataFrame(columns=['Price', 'Quantity', 'Side'])

        ################
        # plot control #
        ################

        self.row_control = dict()
        self.color_control = dict()
        self.color_fill_control = dict()
        self.indicators_filled_mode = dict()
        self.axis_groups = dict()
        self.global_axis_group = 99
        self.strategies = 0
        self.row_counter = 1
        self.strategy_groups = dict()
        self.plot_splitted_serie_couples = {}

        ##############
        # timestamps #
        ##############

        self.hours = hours
        self.start_time, self.end_time = setup_startime_endtime(start_time=start_time,
                                                                end_time=end_time,
                                                                time_zone=self.time_zone,
                                                                hours=self.hours,
                                                                closed=self.closed,
                                                                tick_interval=self.tick_interval,
                                                                limit=self.limit)

        binpan_logger.debug(f"New instance of BinPan Symbol {self.version}: {self.symbol},"
                            f" {self.tick_interval}, limit={self.limit}, start={self.start_time},"
                            f" end={self.end_time}, {self.time_zone}, closed_candles={self.closed}")

        #################
        # query candles #
        #################

        if not from_csv and not self.postgres_klines:
            self.raw = get_candles_by_time_stamps(symbol=self.symbol,
                                                  tick_interval=self.tick_interval,
                                                  start_time=self.start_time,
                                                  end_time=self.end_time,
                                                  limit=self.limit)

            self.df = parse_candles_to_dataframe(raw_response=self.raw,
                                                 columns=self.original_columns,
                                                 time_cols=self.time_cols,
                                                 symbol=self.symbol,
                                                 tick_interval=self.tick_interval,
                                                 time_zone=self.time_zone)
        elif self.postgres_klines:
            import handlers.postgresql as postgresql  # solo para pycharm
            self.table = postgresql.sanitize_table_name(f"{self.symbol.lower()}@kline_{self.tick_interval}")
            self.df = postgresql.get_data_and_parse(cursor=self.cursor_klines,
                                                    table=self.table,
                                                    symbol=self.symbol,
                                                    tick_interval=self.tick_interval,
                                                    time_zone=self.time_zone,
                                                    start_time=self.start_time,
                                                    end_time=self.end_time,
                                                    data_type='kline')
        else:
            self.raw = None

        # update timestamps from data
        self.timestamps = self.get_timestamps()
        self.dates = self.get_dates()
        self.start_time, self.end_time = self.timestamps

        self.len = len(self.df)

        ##################
        # exchange setup #
        ##################

        if not info_dic:  # for loop operations can be passed to avoid api weight overcome
            try:
                self.info_dic = get_info_dic()
            except Exception as e:
                binpan_logger.warning(f"Error trying get_info_dic() from API: {e}")
                self.info_dic = dict()
        else:
            assert type(info_dic) == dict, "info_dic must be a dictionary"
            assert len(info_dic) > 0, "info_dic must be a dictionary with data"
            self.info_dic = info_dic

        try:
            self.tickSize = self.info_dic[self.symbol]['filters'][0]['tickSize']
            self.decimals = get_decimal_positions(self.tickSize)
            self.pip = self.tickSize
            self.order_filters = self.get_order_filters()
            self.order_types = self.get_order_types()
            self.permissions = self.get_permissions()
            self.precision = self.get_precision()
        except KeyError:  # maybe no internet connectio but database available in local
            self.tickSize = None
            self.decimals = None
            self.pip = None
            self.order_filters = None
            self.order_types = None
            self.permissions = None
            self.precision = None

        # self.tickSize = self.info_dic[self.symbol]['filters'][0]['tickSize']
        # self.decimals = get_decimal_positions(self.tickSize)
        # self.pip = self.tickSize
        #
        # self.order_filters = self.get_order_filters()
        # self.order_types = self.get_order_types()
        # self.permissions = self.get_permissions()
        # self.precision = self.get_precision()

        self.market_profile_df = pd.DataFrame()
        self.rolling_support_resistance_df = pd.DataFrame()

        # init vertical lines over candles
        self.red_timestamps = []
        self.blue_timestamps = []

        # check api continuity data and notify to user
        self.discontinuities = check_continuity(df=self.df, time_zone=self.time_zone)

    def __repr__(self):
        return str(self.df)

    ##################
    # Show variables #
    ##################

    def save_csv(self, timestamped_filename: bool = True):
        """
        Saves current csv to a csv file.

        :param bool timestamped_filename: Adds start and end timestamps to the name.
        :return: None
        """
        df_ = self.df
        if timestamped_filename:
            start, end = self.get_timestamps()
            filename = f"{df_.index.name.replace('/', '-')} klines {start} {end}.csv"
        else:
            filename = f"{df_.index.name.replace('/', '-')} klines.csv"

        save_dataframe_to_csv(filename=filename, data=df_, timestamp=not timestamped_filename)
        binpan_logger.info(f"Saved file {filename}")

    def save_atomic_trades_csv(self, timestamped_filename: bool = True):
        """
        Saves current atomic trades to a csv file.

        :param bool timestamped_filename: Adds start and end timestamps to the name.
        :return: None
        """
        if self.atomic_trades.empty:
            binpan_logger.info(f"No atomic trades to save.")
        else:
            df_ = self.atomic_trades
            if timestamped_filename:
                start, end = self.get_timestamps()
                filename = f"{df_.index.name.replace('/', '-')} atomicTrades {start} {end}.csv"
            else:
                filename = f"{df_.index.name.replace('/', '-')} atomicTrades.csv"

            save_dataframe_to_csv(filename=filename, data=df_, timestamp=not timestamped_filename)
            binpan_logger.info(f"Saved file {filename}")

    def save_agg_trades_csv(self, timestamped_filename: bool = True):
        """
        Saves current aggregated trades to a csv file.

        :param bool timestamped_filename: Adds start and end timestamps to the name.
        :return: None
        """
        if self.agg_trades.empty:
            binpan_logger.info(f"No aggregated trades to save.")
        else:
            df_ = self.agg_trades
            if timestamped_filename:
                start, end = self.get_timestamps()
                filename = f"{df_.index.name.replace('/', '-')} aggTrades {start} {end}.csv"
            else:
                filename = f"{df_.index.name.replace('/', '-')} aggTrades.csv"

            save_dataframe_to_csv(filename=filename, data=df_, timestamp=not timestamped_filename)
            binpan_logger.info(f"Saved file {filename}")

    ##################
    # pandas display #
    ##################

    def set_display_columns(self, display_columns=None):
        """
        Method to change the number of columns shown in the display of the dataframe. Uses pandas options.

        :param int display_columns: Integer
        :return: None
        """
        if display_columns:
            self.display_columns = display_columns
            pd.set_option('display.max_columns', display_columns)
        else:
            pd.set_option('display.max_columns', self.display_columns)

    def set_display_min_rows(self, display_min_rows=None):
        """
        Method to change the number of minimum rows shown in the display of the dataframe. Uses pandas options.

        :param int display_min_rows: Integer

        """
        if display_min_rows:
            self.display_min_rows = display_min_rows
            pd.set_option('display.min_rows', display_min_rows)

        else:
            pd.set_option('display.min_rows', self.display_min_rows)

    def set_display_max_rows(self, display_max_rows=None):
        """
        Method to change the number of maximum rows shown in the display of the dataframe. Uses pandas options.

        :param int display_max_rows: Integer

        """
        if display_max_rows:
            self.display_max_rows = display_max_rows
            pd.set_option('display.max_rows', display_max_rows)

        else:
            pd.set_option('display.max_rows', self.display_max_rows)

    def set_display_width(self, display_width: int = None):
        """
        Method to change the width shown in the display of the dataframe. Uses pandas options.

        :param int display_width: Integer

        """
        if display_width:
            self.display_width = display_width
            pd.set_option('display.width', display_width)
        else:
            pd.set_option('display.width', self.display_width)

    @staticmethod
    def set_display_decimals(display_decimals: int):
        """
        Method to change the number of decimals shown in the display of the dataframe. Uses pandas options.

        :param int display_decimals: Integer

        """
        arg = f'%.{display_decimals}f'
        pd.set_option('display.float_format', lambda x: arg % x)

    ###########
    # methods #
    ###########

    def basic(self, exceptions: list = None, actions_col='actions'):
        """
        Shows just a basic selection of columns data in the dataframe.

        :param list exceptions: Columns names to keep.
        :param str actions_col: To keep tags for buy or sell actions.
        :return pd.DataFrame: Pandas DataFrame
        """
        return basic_dataframe(data=self.df, exceptions=exceptions, actions_col=actions_col)

    def drop(self, columns_to_drop=None, inplace=False) -> pd.DataFrame:
        """
        It drops some columns from the dataframe. If columns list not passed, then defaults to the initial columns.

        Can be used when messing with indicators to clean the object.

        :param: list columns: A list with the columns names to drop. If not passed, it defaults to the initial columns that remain
            from when instanced. Defaults to any column but initial ones.
        :param: bool inplace: When true, it drops columns in the object. False just returns a copy without that columns and dataframe
            in the object remains.
        :return pd.DataFrame: Pandas DataFrame with columns dropped.

        """
        # if not columns_to_drop:
        #     columns_to_drop = []
        current_columns = self.df.columns
        if not columns_to_drop:
            columns_to_drop = []
            for col in current_columns:
                if not col in self.original_columns:
                    columns_to_drop.append(col)  # self.row_counter = 1
        try:
            if inplace:
                conserve_columns = [c for c in current_columns if c not in columns_to_drop and c in self.row_control.keys()]
                # conserve_columns = [c for c in current_columns if c not in columns_to_drop]

                # clean strategy groups
                clean_strategy_groups = {}
                for k, v in self.strategy_groups.items():
                    clean_strategy_groups[k] = []
                    for col in v:
                        if col in conserve_columns:
                            clean_strategy_groups[k].append(col)

                clean_strategy_groups = {k: v for k, v in clean_strategy_groups.items() if v}
                self.strategy_groups = clean_strategy_groups

                binpan_logger.debug(f"row_control: {self.row_control}")

                self.row_control = {c: self.row_control[c] for c in conserve_columns}
                extra_rows = set(self.row_control.values())
                try:
                    extra_rows.remove(1)
                except KeyError:
                    pass
                self.row_counter = 1 + len(extra_rows)

                # reacondicionar rows de lo que ha quedado
                unique_rows_index = {v: i + 2 for i, v in enumerate(extra_rows)}
                binpan_logger.debug(f"conserve_columns: {conserve_columns} unique_rows_index: {unique_rows_index} extra_rows: {extra_rows}")

                new_row_control = {}
                for c in conserve_columns:
                    if self.row_control[c] in unique_rows_index.keys():
                        new_row_control.update({c: unique_rows_index[self.row_control[c]]})
                    else:
                        new_row_control.update({c: 1})  # not existing row for a line goes to 1 row
                self.row_control = new_row_control
                # self.row_control = {c: unique_rows_index[self.row_control[c]] if c in self.row_control.keys() else 1 for c in
                # conserve_columns}

                # clean plotting areas info
                self.plot_splitted_serie_couples = {c: self.plot_splitted_serie_couples[c] for c in conserve_columns if
                                                    c in self.plot_splitted_serie_couples.keys()}

                self.color_control = {c: self.color_control[c] for c in conserve_columns}
                self.color_fill_control = {c: self.color_fill_control[c] for c in conserve_columns}

                # revisar esto cuando el fill to next y esté hecho
                self.indicators_filled_mode = {c: self.indicators_filled_mode[c] for c in conserve_columns if
                                               c in self.indicators_filled_mode.keys()}
                self.axis_groups = {c: self.axis_groups[c] for c in conserve_columns if c in self.axis_groups.keys()}

                self.df.drop(columns_to_drop, axis=1, inplace=True)

                return self.df
            else:
                return self.df.drop(columns_to_drop, axis=1, inplace=False)

        except KeyError as exc:
            wrong = (set(self.df.columns) | set(columns_to_drop)) - set(self.df.columns)
            msg = f"BinPan Exception: Wrong column names to drop: {wrong} {exc}"
            binpan_logger.error(msg)
            raise Exception(msg)

    def delete_indicator_family(self, indicator_name_root: str) -> pd.DataFrame or None:
        """
        Deletes indicator from dataframe. It search for plot info and also deletes other indicators in the same plot row level.

        :param str indicator_name_root: Starting characters for the columns to delete. Case-insensitive.
        :return: Resulting main dataframe.
        """
        columns = [c for c in self.df.columns if c.startswith(indicator_name_root)]
        if not columns:
            binpan_logger.debug(f"BinPan Warning: No columns found with root '{indicator_name_root}': {self.df.columns}")
            return
        level = self.row_control[columns[0]]
        columns = self.remove_plot_info_associated_columns(columns=columns, row_level=level)
        for c in columns:
            self.df.drop(c, axis=1, inplace=True)
        return self.df

    def insert_indicator(self,
                         source_data: Union[pd.Series, pd.DataFrame, np.ndarray, list],
                         strategy_group: Optional[str] = None,
                         plotting_row: Optional[int] = None,
                         plotting_rows: Optional[List[int]] = None,
                         color: Optional[str] = None,
                         no_overlapped_plot_rows: bool = True,
                         colors: Optional[List[str]] = None,
                         color_fills: Optional[List[Union[str, bool]]] = None,
                         name: Optional[str] = None,
                         names: Optional[List[str]] = None,
                         suffix: str = '') -> Optional[pd.DataFrame]:
        """
         Adds one or more indicators to the DataFrame in place.

         :param source_data: The source data for the indicator(s). Can be a Series, DataFrame, ndarray, or list thereof.
         :param strategy_group: (Optional) Name of the strategy group to tag the inserted data.
         :param plotting_row: (Optional) The specific row for plotting a single series. '1' overlaps with candles; other values create new rows.
         :param plotting_rows: (Optional) List of rows for plotting each series. '1' means overlap; other integers determine separate row positions.
         :param color: (Optional) Color for plotting a single series.
         :param no_overlapped_plot_rows: If True, avoids overlapping plot rows for multiple series.
         :param colors: (Optional) List of colors for each series indicator. Defaults to random colors if not provided.
         :param color_fills: (Optional) List of color fills (as strings) or False to avoid filling. Example: 'rgba(26,150,65,0.5)'.
         :param name: (Optional) Name for a single inserted object.
         :param names: (Optional) List of names for each column when multiple indicators are inserted.
         :param suffix: Suffix to add to the new column name(s). If the source data is nameless, the suffix becomes the entire name.
         :return: The modified DataFrame with new indicators added, or None if the operation fails.

         Note: This function dynamically assigns plotting rows and colors if they are not explicitly provided. It handles different types of input data for indicators and integrates them into the existing DataFrame.
         """
        if type(source_data) == list:
            data_qty = len(source_data)
        elif type(source_data) == pd.DataFrame:
            data_qty = len(source_data.columns)
        else:
            data_qty = 1

        if plotting_row and not plotting_rows:
            plotting_rows = [plotting_row]
        if not plotting_rows:
            plotting_rows = [2 for _ in range(data_qty)]

        if color and not colors:
            colors = [color]
        if not colors:
            colors = [choice(plotly_colors) for _ in range(data_qty)]

        if not color_fills:
            color_fills = [False for _ in range(data_qty)]

        current_df = self.df.copy(deep=True)

        # get names
        if type(source_data) == pd.Series:
            data = source_data.copy(deep=True)
            data.index = current_df.index
            if name:
                col_name = name
            elif names:
                col_name = names[0]
            elif not data.name:
                if suffix:
                    col_name = suffix
                else:
                    col_name = f'Inserted_{len(self.df.columns)}' + suffix
            else:
                col_name = str(data.name) + suffix

            data.name = col_name
            data_series = [data]

        elif type(source_data) == pd.DataFrame:
            data = source_data.copy(deep=True)
            data.set_index(current_df.index, inplace=True)
            data_series = [data[col] for col in data.columns]
            for name_idx, ser in enumerate(data_series):
                if names:
                    ser.name = names[name_idx]
                else:
                    ser.name = str(ser.name) + suffix

        elif type(source_data) == np.ndarray:
            data = source_data.copy()
            if names:
                data_ser = pd.Series(data=data, index=current_df.index, name=names[0])
            elif suffix:
                data_ser = pd.Series(data=data, index=current_df.index, name=suffix)
            else:
                data_ser = pd.Series(data=data, index=current_df.index, name=f'Inserted_{len(self.df.columns)}')

            del data
            data = data_ser
            data_series = [data_ser]

        elif type(source_data) == list:
            assert len(source_data) == len(plotting_rows)
            # if not colors:
            #     colors = [choice(plotly_colors) for _ in range(len(rows))]
            # if not color_fills:
            #     color_fills = [False for _ in range(len(rows))]

            if name and not names:
                names = [f"{name}_{i}" for i in range(data_qty)]

            for element_idx, new_element in enumerate(source_data):

                assert type(new_element) in [pd.Series, np.ndarray]
                # noinspection PyUnresolvedReferences
                data = new_element.copy(deep=True)

                if not names:
                    try:
                        # noinspection PyUnresolvedReferences
                        current_name = new_element.name
                    except Exception:
                        current_name = f"Indicator_{len(self.df) + element_idx}{suffix}"
                else:
                    current_name = names[element_idx]

                self.insert_indicator(source_data=data, plotting_rows=[plotting_rows[element_idx]], colors=[
                    colors[element_idx]], color_fills=[color_fills[element_idx]], names=[current_name], suffix=suffix)
            return self.df

        else:
            msg = f"BinPan Warning: Unexpected data type {type(source_data)}, expected pd.Series, np.ndarray, pd.DataFrame or list of them."
            binpan_logger.warning(msg)
            return

        if self.is_new(source_data=data, suffix=''):  # suffix is added before this to names
            if no_overlapped_plot_rows:
                # downcast rows to available except 1 (overlap)
                rows_tags = {row: i + self.row_counter + 1 for i, row in enumerate(sorted(list(set(plotting_rows))))}
            else:
                rows_tags = {row: row for row in plotting_rows}

            plotting_rows = [rows_tags[r] if r != 1 else 1 for r in plotting_rows]

            for i, serie in enumerate(data_series):
                column_name = str(serie.name)  # suffix is added before this to names
                current_df.loc[:, column_name] = serie
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=color_fills[i])
                self.set_plot_row(indicator_column=column_name, row_position=plotting_rows[i])

            self.row_counter = max(plotting_rows)

            self.df = current_df

        # tag strategy group for columns
        if strategy_group:
            for d in data_series:
                self.set_strategy_groups(column=str(d.name), group=strategy_group)

        return self.df

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
        df_.index.name += ' HK'

        if inplace:
            self.df = df_
            return self.df
        else:
            return df_

    def get_timestamps(self) -> Tuple[int, int]:
        """
        Get the first Open timestamp and the last Close timestamp.

        :return tuple(int, int): Start Open timestamp and end close timestamp

        """
        if self.df.empty:
            binpan_logger.warning(f"BinPan Warning: Empty dataframe, no timestamps available.")
            return 0, 0
        start = self.df.iloc[0]['Open timestamp']
        # end = self.df.iloc[-1]['Close timestamp']
        end = self.df.iloc[-1]['Open timestamp']
        binpan_logger.debug(f"From first Open timestamp {start} to last Open timestamp {end}")
        return int(start), int(end)

    def get_dates(self) -> Tuple[str, str]:
        """
        Get the first Open timestamp and the last Open timestamp, and converts to timezoned dates.

        :return tuple(int, int): Start Open date and end close date

        """
        start, end = self.get_timestamps()
        ret_start = convert_milliseconds_to_str(ms=start, timezoned=self.time_zone)
        ret_end = convert_milliseconds_to_str(ms=end, timezoned=self.time_zone)
        binpan_logger.debug(f"From first Open date {ret_start} to last Open date {ret_end}")
        return ret_start, ret_end

    def get_agg_trades(self,
                       hours: int = None,
                       minutes: int = None,
                       startTime: int or str = None,
                       endTime: int or str = None,
                       time_zone: str = None,
                       from_csv: str = None) -> pd.DataFrame:
        """
        Calls the API and creates another dataframe included in the object with the aggregated trades from API for the period of the
        created object.

        .. note::

           If the object covers a long time interval, this action can take a relative long time. The BinPan library take care of the
           API weight and can take a sleep to wait until API weight returns to a low value. This avoids ban from the API.

        :param int hours: If passed, it use just last passed hours for the plot.
        :param int minutes: If passed, it use just last passed minutes for the plot.
        :param int or str startTime: If passed, it use just from the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :param int or str endTime: If passed, it use just until the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :param str time_zone: A time zone for time index conversion. Example: "Europe/Madrid"

        Example:

            .. code-block::

                                                  Aggregate tradeId     Price  Quantity  First tradeId  Last tradeId                 Date
                                                       Timestamp  Buyer was maker  Best price match
                BTCBUSD Europe/Madrid
                2022-11-20 14:11:36.763000+01:00          536009627  16524.58   0.21421      632568399     632568399  2022-11-20 14:11:36
                 1668949896763             True              True
                2022-11-20 14:11:36.787000+01:00          536009628  16525.04   0.02224      632568400     632568400  2022-11-20 14:11:36
                 1668949896787             True              True
                2022-11-20 14:11:36.794000+01:00          536009629  16525.04   0.01097      632568401     632568401  2022-11-20 14:11:36
                 1668949896794             True              True
                2022-11-20 14:11:36.849000+01:00          536009630  16525.27   0.05260      632568402     632568403  2022-11-20 14:11:36
                 1668949896849            False              True
                2022-11-20 14:11:36.849000+01:00          536009631  16525.28   0.00073      632568404     632568404  2022-11-20 14:11:36
                 1668949896849            False              True
                ...                                             ...       ...       ...            ...           ...                  ...
                           ...              ...               ...
                2022-11-20 15:10:57.928000+01:00          536083210  16556.75   0.01730      632653817     632653817  2022-11-20 15:10:57
                 1668953457928             True              True
                2022-11-20 15:10:57.928000+01:00          536083211  16556.74   0.00851      632653818     632653819  2022-11-20 15:10:57
                 1668953457928             True              True
                2022-11-20 15:10:57.950000+01:00          536083212  16558.48   0.00639      632653820     632653820  2022-11-20 15:10:57
                 1668953457950            False              True
                2022-11-20 15:10:57.990000+01:00          536083213  16558.48   0.01242      632653821     632653821  2022-11-20 15:10:57
                 1668953457990             True              True
                2022-11-20 15:10:58.020000+01:00          536083214  16558.49   0.00639      632653822     632653822  2022-11-20 15:10:58
                 1668953458020            False              True
                [73588 rows x 9 columns]

        :param str from_csv: If set, loads from a file.
        :return: Pandas DataFrame

        """
        if time_zone:
            self.time_zone = time_zone

        if startTime:
            startTime = convert_str_date_to_ms(date=startTime, time_zone=self.time_zone)
        if endTime:
            endTime = convert_str_date_to_ms(date=endTime, time_zone=self.time_zone)
        if hours:
            startTime = int(time() * 1000) - (1000 * 60 * 60 * hours)
        elif minutes:
            startTime = int(time() * 1000) - (1000 * 60 * minutes)

        if startTime:
            curr_startTime = startTime
        else:
            curr_startTime = self.start_time

        if endTime:
            curr_endTime = endTime
        elif self.end_time:
            curr_endTime = self.end_time
        else:
            curr_endTime = int(time() * 1000)

        st = convert_milliseconds_to_str(curr_startTime, timezoned=self.time_zone)
        en = convert_milliseconds_to_str(curr_endTime, timezoned=self.time_zone)
        print(f"Requesting aggregated trades between {st} and {en}")

        if from_csv:
            if type(from_csv) == str:
                filename = from_csv
            else:
                filename = select_file(path=self.cwd, extension='csv', name_filter='aggTrades')

                # basic metadata
                _, _, _, _, _, _ = extract_filename_metadata(filename=filename,
                                                             expected_data_type="aggTrades",
                                                             expected_symbol=self.symbol,
                                                             expected_timezone=self.time_zone)
            # load and to numeric types
            df_ = read_csv_to_dataframe(filename=filename,
                                        index_col="Timestamp",
                                        secondary_index_col="Aggregate tradeId",
                                        symbol=self.symbol,
                                        index_time_zone=self.time_zone)

            # check columns
            for col in df_.columns:
                if not col in agg_trades_columns_from_binance:
                    raise BinPanException(f"File do not seems to be Aggregated Trades File!")
            self.agg_trades = df_
        elif self.postgres_agg_trades:
            from handlers.postgresql import get_data_and_parse, sanitize_table_name
            agg_table = sanitize_table_name(f"{self.symbol.lower()}_aggTrade")
            self.agg_trades = get_data_and_parse(cursor=self.cursor_agg_trades,
                                                 table=agg_table,
                                                 symbol=self.symbol,
                                                 tick_interval=self.tick_interval,
                                                 time_zone=self.time_zone,
                                                 start_time=curr_startTime,
                                                 end_time=curr_endTime,
                                                 data_type="aggTrade")
        else:
            try:
                self.raw_agg_trades = get_historical_agg_trades(symbol=self.symbol, startTime=curr_startTime, endTime=curr_endTime)
            except Exception as _:
                msg = f"Error fetching raw_agg_trades, maybe missing API key in secret.py file!!!"
                binpan_logger.error(msg)
                self.raw_agg_trades = []

            self.agg_trades = parse_agg_trades_to_dataframe(response=self.raw_agg_trades,
                                                            columns=self.agg_trades_columns,
                                                            symbol=self.symbol,
                                                            time_zone=self.time_zone,
                                                            drop_dupes='Aggregate tradeId')

        self.agg_trades = convert_to_numeric(data=self.agg_trades)

        return self.agg_trades

    def get_atomic_trades(self, hours: int = None,
                          minutes: int = None,
                          startTime: int or str = None,
                          endTime: int or str = None,
                          time_zone: str = None,
                          from_csv: str = None) -> pd.DataFrame:
        """
        Calls the API and creates another dataframe included in the object with the atomic trades from API for the period of the
        created object.

        .. note::

           If the object covers a long time interval, this action can take a relative long time. The BinPan library take care of the
           API weight and can take a sleep to wait until API weight returns to a low value.

        :param int hours: If passed, it use just last passed hours for the plot.
        :param int minutes: If passed, it use just last passed minutes for the plot.
        :param int or str startTime: If passed, it use just from the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :param int or str endTime: If passed, it use just until the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :param str time_zone: A time zone for time index conversion. Example: "Europe/Madrid"
        :param str from_csv: If set, loads from a file.
        :return: Pandas DataFrame

        """
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

        if startTime:
            curr_startTime = startTime
        else:
            curr_startTime = self.start_time

        if endTime:
            curr_endTime = endTime
        elif self.end_time:
            curr_endTime = min(self.end_time, int(time() * 1000))
        else:
            curr_endTime = int(time() * 1000)

        st = convert_milliseconds_to_str(curr_startTime, timezoned=self.time_zone)
        en = convert_milliseconds_to_str(curr_endTime, timezoned=self.time_zone)
        print(f"Requesting atomic trades between {st} and {en}")

        if from_csv:
            if type(from_csv) == str:
                filename = from_csv
            else:
                filename = select_file(path=self.cwd, extension='csv', name_filter='atomicTrades')

            # basic metadata
            _, _, _, _, _, _ = extract_filename_metadata(filename=filename, expected_data_type="atomicTrades",
                                                         expected_symbol=self.symbol, expected_timezone=self.time_zone)

            # load and to numeric types
            df_ = read_csv_to_dataframe(filename=filename,
                                        index_col="Timestamp",
                                        secondary_index_col="Trade Id",
                                        symbol=self.symbol,
                                        index_time_zone=self.time_zone)

            # check columns
            for col in df_.columns:
                if not col in atomic_trades_columns_from_binance:
                    raise BinPanException(f"File do not seems to be Atomic Trades File!")
            self.atomic_trades = df_
        elif self.postgres_atomic_trades:
            from handlers.postgresql import get_data_and_parse
            trade_table = f"{self.symbol.lower()}_trade"

            self.atomic_trades = get_data_and_parse(cursor=self.cursor_atomic_trades,
                                                    table=trade_table,
                                                    symbol=self.symbol,
                                                    tick_interval=self.tick_interval,
                                                    time_zone=self.time_zone,
                                                    start_time=curr_startTime,
                                                    end_time=curr_endTime,
                                                    data_type="trade")
        else:
            try:
                self.raw_atomic_trades = get_historical_atomic_trades(symbol=self.symbol,
                                                                      startTime=curr_startTime,
                                                                      endTime=curr_endTime)
            except Exception:
                msg = f"Error fetching raw_atomic_trades, maybe missing API key in secret.py file!!!"
                binpan_logger.error(msg)
                self.raw_atomic_trades = []

            self.atomic_trades = parse_atomic_trades_to_dataframe(response=self.raw_atomic_trades,
                                                                  columns=self.atomic_trades_columns,
                                                                  symbol=self.symbol,
                                                                  time_zone=self.time_zone,
                                                                  drop_dupes='Trade Id')
        self.atomic_trades = convert_to_numeric(data=self.atomic_trades)
        return self.atomic_trades

    def is_new(self, source_data: pd.Series or pd.DataFrame, suffix: str = '') -> bool:
        """
        Verify if indicator columns are previously created to avoid allocating new rows and colors etc.

        :param pd.Series or pd.DataFrame source_data: Data from pandas_ta to review if is previously computed.
        :param str suffix: If suffix passed, it takes it into account when searching for existence.
        :return bool:
        """
        # existing_columns = list(self.df.columns)

        if type(source_data) == np.ndarray:
            return True

        source_data = source_data.copy(deep=True)

        generated_columns = []

        if type(source_data) == pd.Series:
            serie_name = str(source_data.name) + suffix
            generated_columns.append(serie_name)

        elif type(source_data) == pd.DataFrame:
            generated_columns = [c + suffix for c in list(source_data.columns)]

        else:
            msg = f"BinPan error: (is_new?) source data is not pd.Series or pd.DataFrame"
            binpan_logger.error(msg)
            raise Exception(msg)

        for gen_col in generated_columns:
            if gen_col in self.df.columns:
                binpan_logger.info(f"Existing column: {gen_col} -> No data added to instance.")
                return False
            else:
                binpan_logger.info(f"New column: {gen_col}")

        return True

    def get_reversal_agg_candles(self, min_height: int = 7, min_reversal: int = 4) -> pd.DataFrame or None:
        """
        Resamples aggregated API trades to reversal klines:
           https://atas.net/atas-possibilities/charts/how-to-set-reversal-charts-for-finding-the-market-reversal/

        :param min_height: Defaults to 7. Minimum reversal kline height to close a candle
        :param min_reversal: Defaults to 4. Minimum reversal from hig/low to close a candle
        :return pd.DataFrame: Resample trades to reversal klines. Can be plotted.
        """
        if self.agg_trades.empty:
            binpan_logger.info(empty_agg_trades_msg)
            return

        if min_height:
            self.min_height = min_height
        if min_reversal:
            self.min_reversal = min_reversal

        if self.reversal_agg_klines.empty or min_height or min_reversal:
            self.reversal_agg_klines = reversal_candles(trades=self.agg_trades, decimal_positions=self.decimals,
                                                        time_zone=self.time_zone, min_height=self.min_height,
                                                        min_reversal=self.min_reversal)
        return self.reversal_agg_klines

    def get_reversal_atomic_candles(self, min_height: int = 7, min_reversal: int = 4) -> pd.DataFrame or None:
        """
        Resamples API atomic trades to reversal klines:
           https://atas.net/atas-possibilities/charts/how-to-set-reversal-charts-for-finding-the-market-reversal/

        :param min_height: Defaults to 7. Minimum reversal kline height to close a candle
        :param min_reversal: Defaults to 4. Minimum reversal from hig/low to close a candle
        :return pd.DataFrame: Resample trades to reversal klines. Can be plotted.
        """
        if self.atomic_trades.empty:
            binpan_logger.info(empty_atomic_trades_msg)
            return

        if min_height:
            self.min_height = min_height
        if min_reversal:
            self.min_reversal = min_reversal

        if self.reversal_atomic_klines.empty or min_height or min_reversal:
            self.reversal_atomic_klines = reversal_candles(trades=self.atomic_trades, decimal_positions=self.decimals,
                                                           time_zone=self.time_zone, min_height=self.min_height,
                                                           min_reversal=self.min_reversal)
        return self.reversal_atomic_klines

    def resample(self, tick_interval: str, inplace=False):
        """
        Resample trades to a different tick interval. Tick interval must be higher.
        :param str tick_interval: A binance tick interval. Must be higher than current tick interval.
        :param bool inplace: Change object dataframe permanently whe True is selected. False shows a copy dataframe.
        :return pd.DataFrame: Resampled klines.
        """
        current_index = tick_interval_values.index(self.tick_interval)
        new_index = tick_interval_values.index(tick_interval)

        try:
            assert new_index > current_index
        except Exception as e:
            binpan_logger.error(f"BinPan error: resample must use higher interval: {new_index} not > {current_index} {e}")
            return

        binpan_logger.info(f"Resampling {self.symbol} from {self.tick_interval} to {tick_interval}")
        if inplace:
            self.drop(inplace=True)
            self.tick_interval = tick_interval
            self.row_control = dict()
            self.color_control = dict()
            self.color_fill_control = dict()
            self.indicators_filled_mode = dict()
            self.axis_groups = dict()
            self.global_axis_group = 99
            self.strategies = 0
            self.row_counter = 1
            self.strategy_groups = dict()
            self.plot_splitted_serie_couples = {}
            self.timestamps = self.get_timestamps()
            self.dates = self.get_dates()
            self.start_time, self.end_time = self.timestamps
            self.df = resample_klines(data=self.df, tick_interval=tick_interval)
            self.discontinuities = check_continuity(df=self.df, time_zone=self.time_zone)
            return self.df
        else:
            return resample_klines(data=self.df, tick_interval=tick_interval)

    def repair_continuity(self):
        self.df = repair_kline_discontinuity(df=self.df, time_zone=self.time_zone)
        binpan_logger.info(f"Klines continuity verification after repair")
        # verify discontinuity
        self.discontinuities = check_continuity(df=self.df, time_zone=self.time_zone)
        if self.discontinuities.empty:
            binpan_logger.info(f"Klines continuity OK")

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

    def set_plot_color(self, indicator_column: str = None, color: int or str = None) -> dict:
        """
        Internal control formatting plots. Can be used to change plot color of an indicator.

        :param str indicator_column: column name
        :param color: reassign color to column name
        :return dict: columns with its assigned colors when plotting.

        """
        if indicator_column and color:
            if type(color) == int:
                self.color_control.update({indicator_column: color})
            elif color in plotly_colors:
                self.color_control.update({indicator_column: color})
            else:
                self.color_control.update({indicator_column: choice(plotly_colors)})
        elif indicator_column:
            self.color_control.update({indicator_column: choice(plotly_colors)})
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

        indicators_series = [temp_df[k] for k in self.row_control.keys()]
        indicator_names = [temp_df[k].name for k in self.row_control.keys()]
        indicators_colors = [self.color_control[k] for k in self.row_control.keys()]
        indicators_colors = [c if type(c) == str else plotly_colors[c] for c in indicators_colors]

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
        return candles_tagged(data=temp_df,
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
            return plot_trades(data=managed_data,
                               max_size=max_size,
                               height=height,
                               logarithmic=logarithmic,
                               overlap_prices=overlap_prices,
                               shifted=shifted,
                               title=title)
        else:
            return plot_trades(data=managed_data,
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
            return plot_trades(data=managed_data,
                               max_size=max_size,
                               height=height,
                               logarithmic=logarithmic,
                               overlap_prices=overlap_prices,
                               shifted=shifted,
                               title=title)
        else:
            return plot_trades(data=managed_data,
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

        :param int min_height: It defaults to previous set. Can be reset when plotting.
        :param min_reversal: It defaults to previous set. Can be reset when plotting.
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
            kwargs['title'] = f"Reversal Candles {self.min_height}/{self.min_reversal} {self.symbol}"
        if not 'yaxis_title' in kwargs.keys():
            kwargs['yaxis_title'] = f"Price {self.symbol}"
        if not 'candles_ta_height_ratio' in kwargs.keys():
            kwargs['candles_ta_height_ratio'] = 0.7

        if from_atomic:
            return candles_ta(data=self.reversal_atomic_klines, plot_volume='Quantity', text_index=text_index, **kwargs)
        else:
            return candles_ta(data=self.reversal_agg_klines, plot_volume='Quantity', text_index=text_index, **kwargs)

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
        return plot_pie(serie=self.agg_trades['Quantity'], categories=categories, logarithmic=logarithmic, title=title)

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

        return plot_hists_vs(x0=aggressive_sellers, x1=aggressive_byers, x0_name="Aggressive sellers", x1_name='Aggressive byers',
                             bins=bins, hist_funct=hist_funct, height=height, title=title, **kwargs_update_layout)

    def plot_market_profile(self, bins: int = 100, hours: int = None, minutes: int = None, startTime: int or str = None,
                            endTime: int or str = None, height=900, from_agg_trades=False, from_atomic_trades=False, title: str = None,
                            time_zone: str = None, **kwargs_update_layout):
        """
        Plots volume histogram by prices segregated aggressive buyers from sellers.


        :param int bins: How many bars.
        :param int hours: If passed, it use just last passed hours for the plot.
        :param int minutes: If passed, it use just last passed minutes for the plot.
        :param int or str startTime: If passed, it use just from the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :param int or str endTime: If passed, it use just until the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
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
            title += f' Aggregated Trades'
            _df = self.agg_trades.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            return bar_plot(df=_df, x_col_to_bars='Price', y_col='Quantity', bar_segments='Buyer was maker', split_colors=True,
                            bins=bins, title=title, height=height, y_axis_title='Buy takers VS Buy makers', horizontal_bars=True,
                            **kwargs_update_layout)
        elif from_atomic_trades:
            title += f' Atomic Trades'
            _df = self.atomic_trades.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            return bar_plot(df=_df, x_col_to_bars='Price', y_col='Quantity', bar_segments='Buyer was maker', split_colors=True,
                            bins=bins, title=title, height=height, y_axis_title='Buy takers VS Buy makers', horizontal_bars=True,
                            **kwargs_update_layout)
        else:
            title += f' Klines'
            _df = self.df.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            binpan_logger.info(f"Using klines data. For deeper info add trades data, example: my_symbol.get_agg_trades()")
            # todo: market_profile sacado de las velas en modo melt para plotly
            profile = market_profile_from_klines_melt(df=_df)
            profile.reset_index(inplace=True)
            return bar_plot(df=profile, x_col_to_bars='Market_Profile', y_col='Volume', bar_segments='Is_Maker', split_colors=True,
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
            return plot_scatter(df=data, x_col=x, y_col=y, color=color, marginal=marginal, title=title, height=height, **kwargs)
        else:
            data = self.agg_trades.copy(deep=True)
            if not (type(x) == str and type(y) == str) and type(color):
                x = x[0]
                y = y[0]
                color = color[0]
            title = f"Priced volume for {self.symbol} data obtained from historical trades."
            return plot_scatter(df=data, x_col=x, y_col=y, symbol=dot_symbol, color=color, marginal=marginal, title=title, height=height,
                                **kwargs)

    def plot_orderbook(self, accumulated=True, title='Depth orderbook plot', height=800, plot_y="Quantity", **kwargs):
        """
        Plots orderbook depth.
        """
        if self.orderbook.empty:
            binpan_logger.info("Orderbook not downloaded. Please add orderbook data with: my_binpan.get_orderbook()")
            return
        return orderbook_depth(df=self.orderbook, accumulated=accumulated, title=title, height=height, plot_y=plot_y, **kwargs)

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

        """

        if self.orderbook.empty:
            binpan_logger.info("Orderbook not downloaded. Please add orderbook data with: my_binpan.get_orderbook()")
            return

        if not title:
            title = f"Distribution plot for order book {self.symbol}"

        return dist_plot(df=self.orderbook, x_col=x_col, color=color, bins=bins, histnorm=histnorm, height=height, title=title,
                         **update_layout_kwargs)

    def plot_taker_maker_ratio_profile(self, bins: int = 100, hours: int = None, minutes: int = None, startTime: int or str = None,
                                       endTime: int or str = None, from_agg_trades=False, from_atomic_trades=False, time_zone: str = None,
                                       title: str = "Taker Buy Ratio Profile", height=1200, width=800, **kwargs_update_layout):
        """
        Plots taker vs maker ratio profile.

        :param int bins: How many bars.
        :param int hours: If passed, it use just last passed hours for the plot.
        :param int minutes: If passed, it use just last passed minutes for the plot.
        :param int or str startTime: If passed, it use just from the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
        :param int or str endTime: If passed, it use just until the timestamp or date in format
         (%Y-%m-%d %H:%M:%S: **2022-05-11 06:45:42**)) for the plot.
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
        profile = self.get_taker_maker_ratio_profile(bins=bins, hours=hours, minutes=minutes, startTime=startTime, endTime=endTime,
                                                     from_agg_trades=from_agg_trades, from_atomic_trades=from_atomic_trades,
                                                     time_zone=time_zone)
        return profile_plot(serie=profile, title=title, height=height, width=width, x_axis_title="Price Buckets",
                            y_axis_title="Taker/Maker ratio", vertical_bar=0.5, **kwargs_update_layout)

    #################
    # Exchange data #
    #################
    def update_info_dic(self):
        """
        Returns exchangeInfo data when instantiated. It includes, filters, fees, and many other data for all symbols in the
        exchange.

        :return dict:
        """
        self.info_dic = get_info_dic()
        return self.info_dic

    def get_order_filters(self) -> dict:
        """
        Get exchange info about the symbol for order filters.
        :return dict:
        """
        filters = get_symbols_filters(info_dic=self.info_dic)
        self.order_filters = filters[self.symbol]
        return self.order_filters

    def get_order_types(self) -> dict:
        """
        Get exchange info about the symbol for order types.
        :return dict:
        """
        order_types_precision = get_orderTypes_and_permissions(info_dic=self.info_dic)
        self.order_types = order_types_precision[self.symbol]['orderTypes']
        return self.order_types

    def get_permissions(self) -> dict:
        """
        Get exchange info about the symbol for trading permissions.
        :return dict:
        """
        permissions = get_orderTypes_and_permissions(info_dic=self.info_dic)
        self.permissions = permissions[self.symbol]['permissions']
        return self.permissions

    def get_precision(self) -> dict:
        """
        Get exchange info about the symbol for assets precision.
        :return dict:
        """
        precision = get_precision(info_dic=self.info_dic)
        self.precision = precision[self.symbol]
        return self.precision

    def get_api_status(self) -> str:
        """
        Return the symbol status, TRADING, BREAK, etc.
        """
        return Exchange().df.loc[self.symbol].to_dict()['status']

    ###############
    # Backtesting #
    ###############

    def backtesting(self,
                    actions_col: str or int,
                    target_column: str or pd.Series = None,
                    stop_loss_column: str or pd.Series = None,
                    entry_filter_column: str or pd.Series = None,
                    fixed_target: bool = True,
                    fixed_stop_loss: bool = True,
                    base: float = 0,
                    quote: float = 1000,
                    priced_actions_col: str = 'Open',
                    label_in=1, label_out=-1,
                    fee: float = 0.001,
                    evaluating_quote: str = None,
                    short: bool = False,
                    inplace=True,
                    suffix: str = None,
                    colors: list = None) -> pd.DataFrame or pd.Series:
        """
        Simulates buys and sells using labels in a tagged column with actions. Actions are considered before the tag, in the next
        candle using priced_actions_col price of that candle before.

        :param str or int actions_col: A column name or index.
        :param target_column: Column with data for operation target values.
        :param stop_loss_column: Column with data for operation stop loss values.
        :param pd.Series or str entry_filter_column: A serie or colum with ones or zeros to allow or avoid entries.
        :param bool fixed_target: Target for any operation will be calculated and fixed at the beginning of the operation.
        :param bool fixed_stop_loss: Stop loss for any operation will be calculated and fixed at the beginning of the operation.
        :param float base: Base inverted quantity.
        :param float quote: Quote inverted quantity.
        :param str or int priced_actions_col: Columna name or index with prices to use when action label in a row.
        :param str or int label_in: A label consider as trade in trigger.
        :param str or int label_out: A label consider as trade out trigger.
        :param float fee: Fees applied to the simulation.
        :param str evaluating_quote: A quote used to convert value of the backtesting line for better reference.
        :param bool short: Backtest in short mode, with in as shorts and outs as repays.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: Defaults to red and green.
        :return pd.DataFrame or pd.Series:

        """

        if type(actions_col) == int:
            actions = self.df.iloc[:, actions_col]
        else:
            actions = self.df[actions_col]

        if suffix:
            suffix = '_' + suffix

        if not short:
            wallet_df = backtesting(df=self.df, actions_column=actions, target_column=target_column, stop_loss_column=stop_loss_column,
                                    entry_filter_column=entry_filter_column, priced_actions_col=priced_actions_col,
                                    fixed_target=fixed_target,
                                    fixed_stop_loss=fixed_stop_loss, base=base, quote=quote, fee=fee, label_in=label_in,
                                    label_out=label_out, suffix=suffix,
                                    evaluating_quote=evaluating_quote, info_dic=self.info_dic)
        else:
            wallet_df = backtesting_short(df=self.df, actions_column=actions, target_column=target_column,
                                          stop_loss_column=stop_loss_column, entry_filter_column=entry_filter_column,
                                          priced_actions_col=priced_actions_col,
                                          fixed_target=fixed_target, fixed_stop_loss=fixed_stop_loss, base=base, quote=quote, fee=fee,
                                          label_in=label_in,
                                          label_out=label_out, suffix=suffix, evaluating_quote=evaluating_quote, info_dic=self.info_dic)

        if inplace and self.is_new(wallet_df):
            column_names = wallet_df.columns
            self.row_counter += 1
            if not colors:
                colors = ['cornflowerblue', 'blue', 'black', 'grey', 'green']
            for i, col in enumerate(column_names):
                self.set_plot_color(indicator_column=col, color=colors[i])
                self.set_plot_color_fill(indicator_column=col, color_fill=False)
                self.set_plot_row(indicator_column=col, row_position=self.row_counter + i)

            # second row added in loop, need to sync row counter with las added row
            self.row_counter += 1
            self.df = pd.concat([self.df, wallet_df], axis=1)

        return wallet_df

    def roi(self, column: str = None) -> float:
        """
        It returns win or loos percent for a evaluation column. Just compares first and last value increment by the first price in percent.
        If not column passed, it will search for an Evaluation column.

        :param str column: A column in the BinPan's DataFrame with values to check ROI (return of inversion).
        :return float: Resulting return of inversion.
        """
        if not column:
            column = [i for i in self.df.columns if i.startswith('Eval')][-1]
            print(f"Auto selected column {column}")

        my_column = self.df[column].copy(deep=True)
        my_column.dropna(inplace=True)

        first = my_column.iloc[0]
        last = my_column.iloc[-1]

        return 100 * (last - first) / first

    def profit_hour(self, column: str = None) -> float:
        """
        It returns win or loos quantity per hour. Just compares first and last value. Expected datetime index. If not column passed, it
        will search for an Evaluation column.

        :param str column: A column in the BinPan's DataFrame with values to check profit with expected datetime index.
        :return float: Resulting return of inversion.
        """
        if not column:
            column = [i for i in self.df.columns if i.startswith('Eval')]
            if not column:
                column = "Close"
            else:
                column = column[-1]
            print(f"Auto selected column {column}")

        my_column = self.df[column].copy(deep=True)
        my_column.dropna(inplace=True)

        first = my_column.iloc[0]
        last = my_column.iloc[-1]

        profit = last - first
        ms = self.df['Close timestamp'].dropna().iloc[-1] - self.df['Open timestamp'].dropna().iloc[0]
        hours = ms / (1000 * 60 * 60)

        print(f"Total profit for {column}: {profit} with ratio {profit / hours} per hour.")

        return profit / hours

    ######################
    # More Exchange Data #
    ######################

    def get_fees(self, symbol: str = None):
        """
        Shows applied fees for the symbol of the object.

        Requires API key added. Look for the add_api_key function in the files_and_filters submodule.

        :param symbol: Not to use it, just here for initializing the class.
        :return: Dictionary
        """
        if not symbol:
            symbol = self.symbol
        api_key, api_secret = get_encoded_secrets()

        return get_fees(symbol=symbol, decimal_mode=False, api_key=api_key, api_secret=api_secret)

    def get_orderbook(self, limit: int = 5000) -> pd.DataFrame:
        """
        Gets orderbook.

        :param int limit: Defaults to maximum: 5000
        :return pd.DataFrame:
        """
        orders = get_order_book(symbol=self.symbol, limit=limit)
        bids = pd.DataFrame(data=orders['bids'], columns=['Price', 'Quantity']).astype(float)
        bids.loc[:, 'Side'] = 'bid'
        asks = pd.DataFrame(data=orders['asks'], columns=['Price', 'Quantity']).astype(float)
        asks.loc[:, 'Side'] = 'ask'
        ob = pd.concat([bids, asks]).sort_values(by=['Price'], ascending=False).reset_index(drop=True)
        ob.index.name = f"{self.symbol} updateId:{orders['lastUpdateId']}"
        self.orderbook = ob
        return self.orderbook

    ##############
    # Indicators #
    ##############

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

        # if self.is_numba and ma_name == 'ema':
        if ma_name == 'ema':
            ma_ = ema_numba(df[column_source].values, window=kwargs['length'])
            ma = pd.Series(data=ma_, index=df.index, name=f"EMA_{kwargs['length']}")

        # elif self.is_numba and ma_name == 'sma':
        elif ma_name == 'sma':
            ma_ = sma_numba(df[column_source].values, window=kwargs['length'])
            ma = pd.Series(data=ma_, index=df.index, name=f"SMA_{kwargs['length']}")

        else:
            ma = ta.ma(name=ma_name, source=df[column_source], **kwargs)

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
        if suffix:
            kwargs.update({'suffix': suffix})
        supertrend_df = ta.supertrend(high=self.df['High'], low=self.df['Low'], close=self.df[
            'Close'], length=length, multiplier=int(multiplier), **kwargs)
        supertrend_df.replace(0, np.nan, inplace=True)  # pandas_ta puts a zero at the beginning sometimes that can break the plot scale

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
        macd = self.df.ta.macd(fast=fast, slow=slow, signal=smooth, **kwargs)
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
        stoch_df = ta.stochrsi(close=self.df[
            'Close'], length=rsi_length, rsi_length=rsi_length, k_smooth=k_smooth, d_smooth=d_smooth, **kwargs)
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

        on_balance = ta.obv(close=self.df['Close'], volume=self.df['Volume'], **kwargs)

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

        ad = ta.ad(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], volume=self.df['Volume'], **kwargs)

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

        vwap = ta.vwap(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], volume=self.df['Volume'], anchor=anchor, **kwargs)

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

        atr = ta.atr(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], length=length, **kwargs)

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

        cci = ta.cci(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], length=length, c=scaling, **kwargs)

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

        eom = ta.eom(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], volume=self.df[
            'Volume'], length=length, divisor=divisor, drift=drift, **kwargs)

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
        and the price "n" day’s ago. The indicator fluctuates around the zero line.

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

        roc = ta.roc(close=self.df['Close'], length=length, escalar=escalar, **kwargs)

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
        bbands = self.df.ta.bbands(close=self.df['Close'], length=length, std=std, ddof=ddof, suffix=suffix, **kwargs)

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
        stoch_df = ta.stoch(high=self.df['High'], low=self.df['Low'], close=self.df[
            'Close'], k=k_length, d=stoch_d, k_smooth=k_smooth, **kwargs)
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
        direction. It does this by taking multiple averages and plotting them on a chart. It also uses these figures to compute a “cloud”
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
                            suffix: str = '', colors: list = None) -> Tuple[pd.Series or None, float or None, float or None]:
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
        # Verificar si el atributo 'name' existe en el módulo 'ta'
        if hasattr(ta, name):
            # Obtener el atributo y llamarlo con los argumentos de 'kwargs'
            indicator_func = getattr(ta, name)
            print(f"This indicator should be plotted manually with the plotting module. Check docs for candles_ta functions and pass it "
                  f"as an indicator serie. Example: candles_ta(data=btcusdt.df, indicators_series=[rsi])")
            return indicator_func(**kwargs)
        else:
            raise ValueError(f"Indicator '{name}' not found in the 'pandas_ta' module.")

    def support_resistance(self,
                           from_atomic: bool = False,
                           from_aggregated: bool = False,
                           max_clusters: int = 5,
                           by_quantity: float = None,
                           simple: bool = True,
                           inplace=True,
                           colors: list = None) -> Tuple[List[float], List[float]]:
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
        rolling support and resistance minutes_window and time_steps_minutes. It can be any of the binance kline ones: '1m', '3m', '5m', '15m',
        '30m', '1h', '2h', '4h', etc

        The parameter delayed is useful when you want to calculate the rolling support and resistance with a delay. For example, if you
        want to calculate the rolling support and resistance with the last 5 minutes of data, but you want to project it 5 minutes
        after the last minute of the window, you can pass delayed=1 (1 step of the interval selected). Useful for projecting support and
        resistance levels in the future.

        If simple parameter is True, it will calculate support and resistance levels merged. Just levels. Default is True.

        Example: If you want to calculate the rolling support and resistance with and interval of 24h and a delayed of 1, this will add past 24h
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
            calculate the rolling support and resistance. It can be any of the following: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', etc
        :param bool from_atomic: If True, support and resistance levels will be calculated using atomic trades.
        :param bool from_aggregated: If True, support and resistance levels will be calculated using aggregated trades.
        :param int max_clusters: If passed, fixes count of levels of support and resistance. Default is 5.
        :param float by_quantity: It takes each price into account by how many times the specified quantity appears in "Quantity" column.
        :param bool simple: If True, it will calculate support and resistance levels merged. Just levels. Default is True.
        :param bool inplace: If True, it will replace the current dataframe with the new one. Default is True.
        :param int delayed: If passed, it will project the rolling support and resistance levels in the future. Default is 0  and means 0 windows projected in the future.
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
                       simple: bool = True) -> Tuple[List[float], List[float]]:
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

    #############
    # Relations #
    #############

    def tag(self,
            column: str or int or pd.Series,
            reference: str or int or float or pd.Series,
            relation: str = 'gt',
            match_tag: str or int = 1,
            mismatch_tag: str or int = 0,
            strategy_group: str = '',
            inplace=True, suffix: str = '',
            color: str or int = 'green') -> pd.Series:
        """
        It tags values of a column/serie compared to other serie or value by methods gt,ge,eq,le,lt as condition.

        :param pd.Series or str column: A numeric serie or column name or column index. Default is Close price.
        :param pd.Series or str or int or float reference: A number or numeric serie or column name.
        :param str relation: The condition to apply comparing column to reference (default is greater than):
            eq (equivalent to ==) — equals to
            ne (equivalent to !=) — not equals to
            le (equivalent to <=) — less than or equals to
            lt (equivalent to <) — less than
            ge (equivalent to >=) — greater than or equals to
            gt (equivalent to >) — greater than
        :param int or str match_tag: Value or string to tag matched relation.
        :param int or str mismatch_tag: Value or string to tag mismatched relation.
        :param str strategy_group: A name for a group of columns to assign to a strategy.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :return pd.Series: A serie with tags as values.


        .. code-block::

           import binpan

           sym = binpan.Symbol('btcbusd', '1m')
           sym.ema(window=200, color='darkgrey')

           # comparing close price (default) greater or equal, than exponential moving average of 200 ticks window previously added.
           sym.tag(reference='EMA_200', relation='ge')
           sym.plot()

        .. image:: images/relations/tag.png
           :width: 1000

        """

        if not relation in ['gt', 'ge', 'eq', 'le', 'lt']:
            raise Exception("BinPan Error: relation must be 'gt','ge','eq','le' or 'lt'")

        # parse params
        if type(column) == str:
            data_a = self.df[column]
        elif type(column) == int:
            data_a = self.df.iloc[:, column]
        else:
            data_a = column.copy(deep=True)

        if type(reference) == str:
            data_b = self.df[reference]
        elif type(reference) == int or type(reference) == float:
            data_b = pd.Series(data=reference, index=data_a.index)
        else:
            data_b = reference.copy(deep=True)

        compared = tag_comparison(serie_a=data_a, serie_b=data_b, **{relation: True}, match_tag=match_tag, mismatch_tag=mismatch_tag)

        if not data_b.name:
            data_b.name = reference

        if suffix:
            suffix = '_' + suffix

        column_name = f"Tag_{data_a.name}_{relation}_{data_b.name}" + suffix
        compared.name = column_name

        if inplace and self.is_new(compared):
            self.row_counter += 1

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)  # overlaps are one
            self.df.loc[:, column_name] = compared

        if strategy_group:
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return compared

    def cross(self,
              slow: str or int or float or pd.Series,
              fast: str or int or pd.Series = 'Close',
              cross_over_tag: str or int = 1,
              cross_below_tag: str or int = -1, echo=0,
              non_zeros: bool = True,
              strategy_group: str = None,
              inplace=True,
              suffix: str = '',
              color: str or int = 'green') -> pd.Series:
        """
        It tags crossing values from a column/serie (fast) over a serie or value (slow).

        :param pd.Series or str or int or float slow: A number or numeric serie or column name.
        :param pd.Series or str fast: A numeric serie or column name or column index. Default is Close price.
        :param int or str cross_over_tag: Value or string to tag matched crossing fast over slow.
        :param int or str cross_below_tag: Value or string to tag crossing slow over fast.
        :param bool non_zeros: Result will not contain zeros as non tagged values, instead will be nans.
        :param int echo: It tags a fixed amount of candles forward the crossed point not including cross candle. If echo want to be used,
         must be used non_zeros.
        :param str strategy_group: A name for a group of columns to assign to a strategy.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :return pd.Series: A serie with tags as values. 1 and -1 for both crosses.

        .. code-block::

           import binpan

           sym = binpan.Symbol(symbol='ethbusd', tick_interval='1m', limit=300, time_zone='Europe/Madrid')
           sym.ema(window=10, color='darkgrey')

           sym.cross(slow='Close', fast='EMA_10')

           sym.plot(actions_col='Cross_EMA_10_Close', priced_actions_col='EMA_10',
                            labels=['over', 'below'],
                            markers=['arrow-bar-left', 'arrow-bar-right'],
                            marker_colors=['orange', 'blue'])

        .. image:: images/relations/cross.png
           :width: 1000

        """

        # parse params
        if type(fast) == str:
            data_a = self.df[fast]
        elif type(fast) == int:
            data_a = self.df.iloc[:, fast]
        else:
            data_a = fast.copy(deep=True)

        if type(slow) == str:
            data_b = self.df[slow]
        elif type(slow) == int or type(slow) == float:
            data_b = pd.Series(data=slow, index=data_a.index)
        else:
            data_b = slow.copy(deep=True)

        if not data_a.name:
            data_a.name = fast
        if not data_b.name:
            data_b.name = slow

        if suffix:
            suffix = '_' + suffix

        column_name = f"Cross_{data_b.name}_{data_a.name}" + suffix

        cross = tag_cross(serie_a=data_a, serie_b=data_b, echo=echo, cross_over_tag=cross_over_tag, cross_below_tag=cross_below_tag,
                          name=column_name, non_zeros=non_zeros)

        if inplace and self.is_new(cross):
            self.row_counter += 1
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)  # overlaps are one
            self.df.loc[:, column_name] = cross

        if strategy_group:
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return cross

    def shift(self,
              column: str or int or pd.Series, window=1,
              strategy_group: str = '',
              inplace=True, suffix: str = '',
              color: str or int = 'grey'):
        """
        It shifts a candle ahead by the window argument value (or backwards if negative)

        :param str or int or pd.Series column: Column to shift values.
        :param int window: Number of candles moved ahead.
        :param str strategy_group: A name for a group of columns to assign to a strategy.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :return pd.Series: A serie with tags as values.
        """
        if type(column) == str:
            data_a = self.df[column]
        elif type(column) == int:
            data_a = self.df.iloc[:, column]
        else:
            data_a = column.copy(deep=True)

        if suffix:
            suffix = '_' + suffix
        column_name = f"Shift_{data_a.name}_{window}" + suffix
        shift = shift_indicator(serie=data_a, window=window)
        shift.name = column_name

        if inplace and self.is_new(shift):

            if data_a.name in self.row_control.keys():
                row_pos = self.row_control[data_a.name]
            elif data_a.name in ['High', 'Low', 'Close', 'Open']:
                row_pos = 1
            else:
                self.row_counter += 1
                row_pos = self.row_counter

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=column_name, row_position=row_pos)
            self.df.loc[:, column_name] = shift

        if strategy_group:
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)

        return shift

    def merge_columns(self,
                      main_column: str or int or pd.Series,
                      other_column: str or int or pd.Series,
                      sign_other: dict = None,
                      strategy_group: str = '',
                      inplace=True,
                      suffix: str = '',
                      color: str or int = 'grey'):
        """
        Predominant serie will be filled nans with values, if existing, from the other serie.

        Same kind of index needed.

        :param pd.Series main_column: A serie with nans to fill from other serie.
        :param pd.Series other_column: A serie to pick values for the nans.
        :param dict sign_other: Replace values by a dict for the "other column". Default is: {1: -1}
        :param str strategy_group: A name for a group of columns to assign to a strategy.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :return pd.Series: A merged serie.
        """
        if not sign_other:
            sign_other = {1: -1}
        if type(main_column) == str:
            data_a = self.df[main_column]
        elif type(main_column) == int:
            data_a = self.df.iloc[:, main_column]
        else:
            data_a = main_column.copy(deep=True)

        if type(other_column) == str:
            data_b = self.df[other_column]
        elif type(other_column) == int:
            data_b = self.df.iloc[:, other_column]
        else:
            data_b = other_column.copy(deep=True)

        if sign_other:
            data_b = data_b.replace(sign_other)

        merged = merge_series(predominant=data_a, other=data_b)
        if suffix:
            suffix = '_' + suffix
        column_name = f"Merged_{data_a.name}_{data_b.name}" + suffix
        merged.name = column_name

        if inplace and self.is_new(merged):

            if data_a.name in self.row_control.keys():
                row_pos = self.row_control[data_a.name]
            elif data_a.name in ['High', 'Low', 'Close', 'Open']:
                row_pos = 1
            else:
                self.row_counter += 1
                row_pos = self.row_counter

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=column_name, row_position=row_pos)
            self.df.loc[:, column_name] = merged

        if strategy_group:
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return merged

    def clean_in_out(self,
                     column: str or int or pd.Series,
                     in_tag=1,
                     out_tag=-1,
                     strategy_group: str = '',
                     inplace=True, suffix: str = '',
                     color: str or int = 'grey'):
        """
        It cleans a serie with in and out tags by eliminating in streaks and out streaks.

        Same kind of index needed.

        :param pd.Series column: A column to clean in and out values.
        :param in_tag: Tag for in tags. Default is 1.
        :param out_tag: Tag for out tags. Default is -1.
        :param str strategy_group: A name for a group of columns to assign to a strategy.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :return pd.Series: A merged serie.
        """
        if type(column) == str:
            data_a = self.df[column]
        elif type(column) == int:
            data_a = self.df.iloc[:, column].copy(deep=True)
        else:
            data_a = column.copy(deep=True)

        clean = clean_in_out(serie=data_a, in_tag=in_tag, out_tag=out_tag)
        if suffix:
            suffix = '_' + suffix
        column_name = f"Clean_{data_a.name}" + suffix

        clean.name = column_name

        if inplace and self.is_new(clean):

            if data_a.name in self.row_control.keys():
                row_pos = self.row_control[data_a.name]
            elif data_a.name in ['High', 'Low', 'Close', 'Open']:
                row_pos = 1
            else:
                self.row_counter += 1
                row_pos = self.row_counter

            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=column_name, row_position=row_pos)
            self.df.loc[:, column_name] = clean

        if strategy_group:
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return clean

    def set_strategy_groups(self, column: str, group: str, strategy_groups: dict = None):
        """
        Returns strategy_groups for BinPan DataFrame.

        :param str column: A column to tag with a strategy group.
        :param str group: Name of the group.
        :param str strategy_groups: The existing strategy groups.
        :return dict: Updated strategy groups of columns.
        """
        if not strategy_groups:
            strategy_groups = self.strategy_groups
        if column and group:
            self.strategy_groups = tag_column_to_strategy_group(column=column, group=group, strategy_groups=strategy_groups)
        return self.strategy_groups

    def get_strategy_columns(self) -> list:
        """
        Returns column names starting with "Strategy".

        :return dict: Updated strategy groups of columns.
        """
        return [i for i in self.df.columns if i.lower().startswith('strategy')]

    def strategy_from_tags_crosses(self,
                                   columns: list = None,
                                   strategy_group: str = '',
                                   matching_tag=1,
                                   method: str = 'all',
                                   tag_reversed_match: bool = False,
                                   inplace=True,
                                   suffix: str = '',
                                   color: str or int = 'magenta',
                                   reversed_match=-1):
        """
        Checks where all tags and cross columns get value "1" at the same time. And also gets points where all tags gets value of "0" and
        cross columns get "-1" value.

        :param list columns: A list of Tag and Cross columns with numeric o 1,0 for tags and 1,-1 for cross points.
        :param str strategy_group: A name for a group of columns to restrict application of strategy. If both columns and strategy_group
         passed, a interjection between the two arguments is applied.
        :param bool tag_reversed_match: If enabled, all zeros or minus ones tag and cross columns are interpreted as reversed match,
         this will enable tagging those.
        :param any matching_tag: A tag to search for the strategy where will be revised method for matched rows.
        :param str method: Can be 'all' or 'any'. It produces a match when all or any columns are matching tags.
        :param any reversed_match: A tag for the all/any not matched strategy rows.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :return pd.Series: A serie with "1" value where all columns are ones and "-1" where all columns are minus ones.
        """
        if columns:
            my_columns = columns
            cross_columns = [c for c in self.df.columns if c.lower().startswith('cross_')]  # used to keep out zeros
        else:
            tag_columns = [c for c in self.df.columns if c.lower().startswith('tag_')]
            cross_columns = [c for c in self.df.columns if c.lower().startswith('cross_')]
            my_columns = tag_columns + cross_columns

        if strategy_group:
            set_my_cols = set(my_columns)
            set_strategy_group = set(self.strategy_groups[strategy_group])
            if columns:
                my_columns = list(set_my_cols.intersection(set_strategy_group))
            else:
                my_columns = self.strategy_groups[strategy_group]
                cross_columns = [c for c in my_columns if c.lower().startswith('cross_')]

        for col in my_columns:
            data_col = self.df[col].dropna()
            try:
                unique_values = data_col.value_counts().index
                numeric_Values = [i for i in unique_values if type(i) in [int, float, complex]]
                assert len(unique_values) == len(numeric_Values)
            except AssertionError:
                raise Exception(f"BinPan Strategic Exception: Not numerica labels on {col}: {list(data_col.value_counts().index)}")

        temp_df = self.df.copy(deep=True)
        temp_df = temp_df.loc[:, my_columns]

        # remove zeros from cross columns
        temp_df[cross_columns] = temp_df[cross_columns].replace({'0': np.nan, 0: np.nan})

        # matching magic
        if method == 'all':
            bull_serie = (temp_df > 0).all(axis=1)
        elif method == 'any':
            bull_serie = (temp_df > 0).any(axis=1)
        else:
            raise Exception(f"BinPan Strategy Exception: Method not 'all' or 'any' -> {method}")

        ret = pd.Series(matching_tag, index=bull_serie[bull_serie].index)

        if tag_reversed_match:
            if method == 'all':
                bear_serie = (temp_df <= 0).all(axis=1)
            elif method == 'any':
                bear_serie = (temp_df <= 0).any(axis=1)
            else:
                raise Exception(f"BinPan Strategy Exception: Method not 'all' or 'any' -> {method}")

            ret_reversed = pd.Series(reversed_match, index=bear_serie[bear_serie].index)
            ret = pd.concat([ret, ret_reversed]).sort_index()

        if suffix:
            suffix = '_' + suffix

        self.strategies += 1
        column_name = f"Strategy_cross_tag_{self.strategies}" + suffix
        ret.name = column_name

        if inplace and self.is_new(ret):
            self.row_counter += 1
            row_pos = self.row_counter
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=column_name, row_position=row_pos)
            self.df.loc[:, column_name] = ret

        return ret

    def ffill_window(self,
                     column: str or int or pd.Series,
                     window: int = 1,
                     inplace=True,
                     replace=False,
                     suffix: str = '',
                     color: str or int = 'blue'):
        """
        It forward fills a value through nans a window ahead.

        :param str or int or pd.Series column: A pandas Series.
        :param int window: Times values are shifted ahead. Default is 1.
        :param bool replace: Permanent replace for a column with results.
        :param bool inplace: Permanent or not. Default is false, because of some testing required sometimes.
        :param str suffix: A string to decorate resulting Pandas series name.
        :param str or int color: A color from plotly list of colors or its index in that list.
        :return pd.Series: A series with index adjusted to the new shifted positions of values.
        """
        if type(column) == str:
            serie = self.df[column]
        elif type(column) == int:
            serie = self.df.iloc[:, column]
        else:
            serie = column.copy(deep=True)

        my_ffill = ffill_indicator(serie=serie, window=window)

        if suffix:
            suffix = '_' + suffix

        self.strategies += 1
        column_name = f"Ffill_{serie.name}_{self.strategies}" + suffix

        my_ffill.name = column_name
        if replace:
            self.df.loc[:, serie.name] = my_ffill

        if inplace and self.is_new(my_ffill):
            self.row_counter += 1
            row_pos = self.row_counter
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=column_name, row_position=row_pos)
            self.df.loc[:, column_name] = my_ffill
        return my_ffill
