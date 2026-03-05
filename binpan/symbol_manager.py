"""

This is the main classes file.

"""
__version__ = "0.8.14"

import os
from sys import path
import pandas as pd

from random import choice
from time import time
import numpy as np

from binpan.exchange_manager import Exchange
from .auxiliar import csv_klines_setup, check_continuity, repair_kline_discontinuity

from handlers.exceptions import BinPanException
from handlers.exchange import (get_decimal_positions, get_info_dic, get_precision, get_orderTypes_and_permissions, get_fees,
                               get_symbols_filters)
from handlers.files import select_file, read_csv_to_dataframe, save_dataframe_to_csv, extract_filename_metadata, get_encoded_secrets

from handlers.indicators import (df_splitter, reversal_candles)

from handlers.logs import LogManager

from handlers.market import (get_candles_by_time_stamps, parse_candles_to_dataframe, convert_to_numeric, basic_dataframe,
                             get_historical_agg_trades, parse_agg_trades_to_dataframe, get_historical_atomic_trades,
                             parse_atomic_trades_to_dataframe, get_order_book)

# handlers.plotting se importa lazy para evitar cargar plotly+scipy al importar binpan
def _plotting():
    from handlers import plotting
    return plotting

from handlers.time_helper import (check_tick_interval, convert_milliseconds_to_str, tick_interval_values)


# handlers.aggregations se importa lazy para evitar cargar scipy al importar binpan

from handlers.standards import (binance_api_candles_cols, agg_trades_api_map_columns, atomic_trades_api_map_columns, time_cols,
                                dts_time_cols, reversal_columns_order, agg_trades_columns_from_binance, atomic_trades_columns_from_binance)

from handlers.wallet import convert_str_date_to_ms
import warnings

# handlers.numba_tools se importa lazy para evitar cargar numba al importar binpan


from objects.timeframes import Timeframe

binpan_logger = LogManager(filename='./logs/binpan.log', name='binpan', info_level='INFO')
version = __version__

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 250)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 12)

empty_agg_trades_msg = "Empty trades, please request using: get_agg_trades() method: Example: my_symbol.get_agg_trades()"
empty_atomic_trades_msg = "Empty atomic trades, please request using: get_atomic_trades() method: Example: my_symbol.get_atomic_trades()"


from .indicators_mixin import IndicatorsMixin
from .plotting_mixin import PlottingMixin
from .strategy_mixin import StrategyMixin


class Symbol(IndicatorsMixin, PlottingMixin, StrategyMixin):
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

    :param bool closed:      The last candle is a closed one in the moment of the creation, discarding the current running one not closed
    yet. Default True.
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
    :param float hours:    If hours is passed, it gets the candles from the last hours.
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
                 hours: float = None,
                 postgres_klines: bool or str = False,
                 postgres_agg_trades: bool or str = False,
                 postgres_atomic_trades: bool or str = False,
                 display_columns: int = 25,
                 display_max_rows: int = 10,
                 display_min_rows: int = 25,
                 display_width: int = 320,
                 from_csv: bool | str = False,
                 info_dic: dict = None):

        if not symbol and not from_csv:
            raise BinPanException(f"BinPan Exception: symbol needed or CSV file to load data using from_csv='filename.csv'")

        if not from_csv and not symbol.isalnum():
            binpan_logger.error(f"BinPan Exception: Ilegal characters in symbol: {symbol}")

        # check correct tick interval passed
        # if not from_csv:
        # tick_interval = check_tick_interval(tick_interval)

        # inicialización temprana para from_csv (se sobreescriben luego)
        self.tick_interval = tick_interval
        self.time_zone = time_zone

        # dataframe columns
        self.original_columns = binance_api_candles_cols
        self.agg_trades_columns = agg_trades_api_map_columns
        self.atomic_trades_columns = atomic_trades_api_map_columns
        self.reversal_columns = reversal_columns_order
        # time cols
        self.time_cols = time_cols
        self.dts_time_cols = dts_time_cols

        # version
        self.version = __version__

        # set de rutas
        self.cwd = os.getcwd()
        path.append(self.cwd)

        # parámetros principales
        self.symbol = symbol.upper() if symbol else ""

        if from_csv:
            (self.df,
             self.symbol,
             self.tick_interval,
             self.time_zone,
             self.start_time,
             self.end_time,
             self.closed,
             self.limit) = csv_klines_setup(from_csv=from_csv,
                                            symbol=self.symbol,
                                            tick_interval=self.tick_interval,
                                            cwd=self.cwd,
                                            time_zone=self.time_zone)
        else:
            ##############
            # timestamps #
            ##############
            tick_interval = check_tick_interval(tick_interval) if tick_interval else None
            self.closed = closed

            # Si hours o start+end definen el rango, ignorar el limit por defecto (1000)
            if hours is not None or (start_time is not None and end_time is not None):
                limit = None

            self.timeframe = Timeframe(start=start_time,
                                       end=end_time,
                                       timezone_IANA=time_zone,
                                       tick_interval=tick_interval,
                                       limit=limit,
                                       hours=hours,
                                       closed=closed)

            self.start_time = self.timeframe.start.open
            self.end_time = self.timeframe.end.close
            self.tick_interval = self.timeframe.tick_interval
            self.limit = self.timeframe.get_limit()
            self.hours = self.timeframe.get_hours()
            self.time_zone = str(self.timeframe.timezone_IANA)

            # self.start_time, self.end_time = setup_startime_endtime(start_time=start_time,
            #                                                         end_time=end_time,
            #                                                         time_zone=self.time_zone,
            #                                                         hours=self.hours,
            #                                                         closed=self.closed,
            #                                                         tick_interval=self.tick_interval,
            #                                                         limit=self.limit)

            # database
            if postgres_klines or postgres_agg_trades or postgres_atomic_trades:

                import handlers.postgresql as postgresql
                from secret import (postgresql_port, postgresql_user, postgresql_database)

                if postgres_klines:
                    if type(postgres_klines) == str:
                        binpan_logger.info(f"Postgres connection requested: {postgres_klines}")
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
        self.display_max_columns = display_columns
        # self.display_min_columns = 10
        self.display_max_rows = display_max_rows
        self.display_min_rows = display_min_rows
        self.display_width = display_width
        self.set_display_max_columns(self.display_max_columns)
        # self.set_display_min_columns(self.display_min_columns)  # no existe este método en pandas
        self.set_display_width(self.display_width)
        self.set_display_min_rows(self.display_min_rows)
        self.set_display_max_rows(self.display_max_rows)

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
        self.plotting_volume_ma = 21
        self.indicators_filled_mode = dict()
        self.axis_groups = dict()
        self.global_axis_group = 99
        self.strategies = 0
        self.row_counter = 1
        self.strategy_groups = dict()
        self.plot_splitted_serie_couples = {}

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

    def set_display_max_columns(self, display_columns=None):
        """
        Method to change the maximum number of columns shown in the display of the dataframe. Uses pandas options.

        :param int display_columns: Integer
        :return: None
        """
        if display_columns:
            self.display_max_columns = display_columns
            pd.set_option('display.max_columns', display_columns)
        else:
            pd.set_option('display.max_columns', self.display_max_columns)

    # def set_display_min_columns(self, display_min_columns=None):
    # no existe este método en pandas
    #     """
    #     Method to change the minimum number of columns shown in the display of the dataframe. Uses pandas options.
    #
    #     :param int display_min_columns: Integer
    #     :return: None
    #     """
    #     if display_min_columns:
    #         self.display_min_columns = display_min_columns
    #         pd.set_option('display.min_columns', display_min_columns)
    #     else:
    #         pd.set_option('display.min_columns', self.display_min_columns)

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
                         source_data: pd.Series | pd.DataFrame | np.ndarray | list,
                         strategy_group: str | None = None,
                         plotting_row: int | None = None,
                         plotting_rows: list[int] | None = None,
                         color: str | None = None,
                         no_overlapped_plot_rows: bool = True,
                         colors: list[str] | None = None,
                         color_fills: list[str | bool] | None = None,
                         name: str | None = None,
                         names: list[str] | None = None,
                         suffix: str = '') -> pd.DataFrame | None:
        """
         Adds one or more indicators to the DataFrame in place.

         :param source_data: The source data for the indicator(s). Can be a Series, DataFrame, ndarray, or list thereof.
         :param strategy_group: (Optional) Name of the strategy group to tag the inserted data.
         :param plotting_row: (Optional) The specific row for plotting a single series. '1' overlaps with candles; other values create
         new rows.
         :param plotting_rows: (Optional) List of rows for plotting each series. '1' means overlap; other integers determine separate row
         positions.
         :param color: (Optional) Color for plotting a single series.
         :param no_overlapped_plot_rows: If True, avoids overlapping plot rows for multiple series.
         :param colors: (Optional) List of colors for each series indicator. Defaults to random colors if not provided.
         :param color_fills: (Optional) List of color fills (as strings) or False to avoid filling. Example: 'rgba(26,150,65,0.5)'.
         :param name: (Optional) Name for a single inserted object.
         :param names: (Optional) List of names for each column when multiple indicators are inserted.
         :param suffix: Suffix to add to the new column name(s). If the source data is nameless, the suffix becomes the entire name.
         :return: The modified DataFrame with new indicators added, or None if the operation fails.

         Note: This function dynamically assigns plotting rows and colors if they are not explicitly provided. It handles different types
         of input data for indicators and integrates them into the existing DataFrame.
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
            colors = [choice(_plotting().plotly_colors) for _ in range(data_qty)]

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
            #     colors = [choice(_plotting().plotly_colors) for _ in range(len(rows))]
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

    def get_timestamps(self) -> tuple[int, int]:
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

    def get_dates(self) -> tuple[str, str]:
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

    def get_atomic_trades(self,
                          hours: int = None,
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
            from handlers.aggregations import resample_klines
            self.df = resample_klines(data=self.df, tick_interval=tick_interval)
            self.discontinuities = check_continuity(df=self.df, time_zone=self.time_zone)
            return self.df
        else:
            from handlers.aggregations import resample_klines
            return resample_klines(data=self.df, tick_interval=tick_interval)

    def repair_continuity(self):
        self.df = repair_kline_discontinuity(df=self.df, time_zone=self.time_zone)
        binpan_logger.info(f"Klines continuity verification after repair")
        # verify discontinuity
        self.discontinuities = check_continuity(df=self.df, time_zone=self.time_zone)
        if self.discontinuities.empty:
            binpan_logger.info(f"Klines continuity OK")

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

