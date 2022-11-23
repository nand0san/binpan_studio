"""

This is the main classes file.

"""
import os
from sys import path

import pandas as pd
import numpy as np

from redis import StrictRedis
from typing import Tuple

# from typing import Tuple

import handlers.exceptions
import handlers.exchange
import handlers.files
import handlers.indicators
import handlers.logs
import handlers.market
import handlers.messages
import handlers.plotting
import handlers.quest
import handlers.redis_fetch
import handlers.starters
import handlers.strategies
import handlers.time_helper
import handlers.wallet

import pandas_ta as ta
from random import choice
import importlib
from time import time

binpan_logger = handlers.logs.Logs(filename='./logs/binpan.log', name='binpan', info_level='INFO')
tick_seconds = handlers.time_helper.tick_seconds
pandas_freq_tick_interval = handlers.time_helper.pandas_freq_tick_interval

__version__ = "0.2.39"

try:
    from secret import redis_conf, redis_conf_trades
except:
    msg = "REDIS: Redis configuration not found in secret.py, if needed, must be passed latter to client. Needed redis server for candles " \
          "and redis server for trades configuration."
    binpan_logger.info(msg)
    pass

try:
    from secret import api_key, api_secret
except ImportError:
    api_key, api_secret = "PLEASE ADD API KEY", "PLEASE ADD API SECRET"
    msg = """\n\n-------------------------------------------------------------
WARNING: No API Key or API Secret

API key would be needed for personal API calls. Any other calls will work.

Adding:

from binpan import handlers

handlers.files.add_api_key("xxxx")
handlers.files.add_api_secret("xxxx")

API keys will be added to a file called secret.py in an encrypted way. API keys in memory stay encrypted except in the API call instant.

Create API keys: https://www.binance.com/en/support/faq/360002502072
"""
    binpan_logger.warning(msg)

plotly_colors = handlers.plotting.plotly_colors

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 250)
pd.set_option('display.min_rows', 10)


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
    :param bool or StrictRedis from_redis: If enabled, BinPan will look for a secret file with the redis ip, port and any other parameter in a map.
       But can be passed a StrictRedis client object previously configured.
       secret.py file map example: redis_conf = {'host':'192.168.1.5','port': 6379,'db': 0,'decode_responses': True}

    :param int display_columns:     Number of columns in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param int display_rows:        Number of rows in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param int display_width:       Display width in the dataframe display. Convenient to adjust in jupyter notebooks.
    :param bool or str from_csv:    If True, gets data from a csv file by selecting interactively from csv files found.
     Also a string with filename can be used.

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


    Created objects contain different instantiated variables like **mysymbol.df** that shows the candles dataframe:

    - **mysymbol.df**: shows candles dataframe
    - **mysymbol.trades**: a pandas dataframe (if requested) with aggregated trades between start and end of the dataframe timestamps.
    - **mysymbol.version**: Version of BinPan.
    - **mysymbol.symbol**: symbol instantiated.
    - **mysymbol.fees**: personal fees applied for the symbol.
    - **mysymbol.tick_interval**: tick_interval selected.
    - **mysymbol.start_time**: start time instantiated.
    - **mysymbol.end_time**: end time instantiated.
    - **mysymbol.limit**: limit of candles in the instance, but if instantiated with start and end times, can be overridden.
    - **mysymbol.time_zone**: timezone of the dates in the index of the dataframe.
    - **mysymbol.time_index**: time index if true or integer index if false.
    - **mysymbol.closed**: asked for dropping not closed candles.
    - **mysymbol.start_ms_time**: timestamp obtained from api in the first candle.
    - **mysymbol.end_ms_time**: timestamp obtained from api in the last candle.
    - **mysymbol.display_columns**: display columns in shell
    - **mysymbol.display_rows**: display rows in shell
    - **mysymbol.display_min_rows**: display max_rows in shell
    - **mysymbol.display_width**: display width in shell
    - **mysymbol.orderbook**: a pandas dataframe (if requested) with last orderbook requested.
    - **mysymbol.row_control**: dictionary with data about plotting control. Represents each dataframe colum position in the plots.
    - **mysymbol.color_control**: dictionary with data about plotting control. Represents each dataframe colum color in the plots.
    - **mysymbol.color_fill_control**: dictionary with data about plotting control. Represents each dataframe colum with color filled to
      zero line in the plots.
    - **mysymbol.indicators_filled_mode**: dictionary with filling mode for each line. Values can be None, tonexty, tozeroy.
    - **mysymbol.axis_group**: dictionary with axis group each line. Values can be None or y, y2, etc.
    - **mysymbol.row_counter**: counter for the indicator rows in a plot.
    - **mysymbol.len**: length of the dataframe
    - **mysymbol.raw**: api klines raw response when instantiated.
    - **mysymbol.info_dic**: exchangeInfo data when instantiated. It includes, filters, fees, and many other data for all symbols in the
      exchange.
    - **mysymbol.order_filters**: filters applied for the symbol when ordering.
    - **mysymbol.order_types**: list of type of orders available for that symbol.
    - **mysymbol.permissions**: list of possible trading ways, like SPOT or MARGIN.
    - **mysymbol.precision**: decimals quantity applied for base and quote assets.

    """

    def __init__(self,
                 symbol: str = None,
                 tick_interval: str = None,
                 start_time: int or str = None,
                 end_time: int or str = None,
                 limit: int = 1000,
                 time_zone: str = 'UTC',
                 time_index: bool = True,
                 closed: bool = True,
                 from_redis: bool or StrictRedis = None,
                 from_redis_trades: bool or StrictRedis = None,
                 display_columns=25,
                 display_rows=10,
                 display_min_rows=25,
                 display_width=320,
                 from_csv: bool or str = False):

        try:
            secret_module = importlib.import_module('secret')
            importlib.reload(secret_module)
            self.api_key = secret_module.api_key
            self.api_secret = secret_module.api_secret
        except ImportError:
            print(f"Binance Api key or Api Secret not found.")
            self.api_key = "INSERT API KEY"
            self.api_secret = "INSERT API KEY"
        except KeyError:
            print(f"Binance Api key or Api Secret not found.")
            self.api_key = "INSERT API KEY"
            self.api_secret = "INSERT API KEY"

        if not symbol and not from_csv:
            raise Exception(f"BinPan symbol Error: symbol needed")

        if not from_csv and not symbol.isalnum():
            binpan_logger.error(f"BinPan error: Ilegal characters in symbol.")

        # check correct tick interval passed
        if not from_csv:
            tick_interval = handlers.time_helper.check_tick_interval(tick_interval)

        self.cwd = os.getcwd()
        path.append(self.cwd)

        self.original_candles_cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote volume',
                                      'Trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore', 'Open timestamp',
                                      'Close timestamp']

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
        if from_csv:
            if type(from_csv) == str:
                filename = from_csv
            else:
                filename = handlers.files.select_file(path=self.cwd, extension='csv')

            binpan_logger.info(f"Loading {filename}")

            # load and to numeric types
            df_ = handlers.files.read_csv_to_dataframe(filename=filename)
            df_ = handlers.market.convert_to_numeric(data=df_)
            # for col in df_.columns:
            #     df_[col] = pd.to_numeric(arg=df_[col], downcast='integer', errors='ignore')

            # basic metadata
            filename = str(os.path.basename(filename))
            symbol = filename.split()[0]
            self.symbol = symbol
            tick_interval = filename.split()[1]
            self.tick_interval = tick_interval
            start_time = int(filename.split()[3])
            end_time = int(filename.split()[4].split('.')[0])
            time_zone = filename.split()[2].replace('-', '/')
            index_name = f"{symbol} {tick_interval} {time_zone}"
            self.time_zone = time_zone

            if time_index:
                df_.sort_values('Open timestamp', inplace=True)
                idx = pd.DatetimeIndex(pd.to_datetime(df_['Open timestamp'], unit='ms')).tz_localize('UTC').tz_convert('Europe/Madrid')
                df_.index = idx
                df_ = df_.asfreq(pandas_freq_tick_interval[tick_interval])  # this adds freq, infer will not work if API blackout period
                self.df = df_
                # drops API blackout periods, but throw errors upwards with some indicators because freq=None before drop.
                # self.df.dropna(how='all', inplace=True)
                self.time_index = True

            self.df.index.name = index_name
            self.limit = len(self.df)

            if from_csv:
                last_timestamp_ind_df = self.df.iloc[-1]['Close timestamp']
                if last_timestamp_ind_df >= int(time() * 1000):
                    self.closed = False
                else:
                    self.closed = True
            else:
                self.closed = closed

        else:
            self.symbol = symbol.upper()
            self.tick_interval = tick_interval
            self.start_time = start_time
            self.end_time = end_time
            self.limit = limit
            self.time_zone = time_zone
            self.time_index = time_index
            self.closed = closed

        self.fees = self.get_fees(symbol=self.symbol)

        if from_redis:
            if type(from_redis) == bool:
                try:
                    self.from_redis = redis_client(**redis_conf)
                except Exception as exc:
                    msg = f"BinPan error: Redis parameters misconfiguration in secret.py -> {exc}"
                    binpan_logger.warning(msg)
                    raise Exception(msg)
            else:
                self.from_redis = from_redis
        else:
            self.from_redis = from_redis

        if from_redis_trades:
            if type(from_redis_trades) == bool:
                try:
                    self.from_redis_trades = redis_client(**redis_conf_trades)
                except Exception as exc:
                    msg = f"BinPan error: Redis trades parameters misconfiguration in secret.py -> {exc}"
                    binpan_logger.warning(msg)
                    raise Exception(msg)
            else:
                self.from_redis_trades = from_redis_trades
        else:
            self.from_redis_trades = from_redis_trades

        self.display_columns = display_columns
        self.display_rows = display_rows
        self.display_min_rows = display_min_rows
        self.display_width = display_width

        self.raw_trades = []
        self.trades = pd.DataFrame(columns=list(self.trades_columns.values()))

        self.orderbook = pd.DataFrame(columns=['Price', 'Quantity', 'Side'])
        self.row_control = dict()
        self.color_control = dict()
        self.color_fill_control = dict()
        self.indicators_filled_mode = dict()
        self.axis_groups = dict()
        # self.action_labels = dict()
        self.global_axis_group = 99
        self.strategies = 0
        self.row_counter = 1
        self.strategy_groups = {}

        self.set_display_columns()
        self.set_display_width()
        self.set_display_min_rows()
        self.set_display_max_rows()

        binpan_logger.debug(f"New instance of BinPan Symbol {self.version}: {self.symbol}, {self.tick_interval}, limit={self.limit},"
                            f" start={self.start_time}, end={self.end_time}, {self.time_zone}, time_index={self.time_index}"
                            f", closed_candles={self.closed}")

        ##############
        # timestamps #
        ##############

        start_time = handlers.wallet.convert_str_date_to_ms(date=start_time, time_zone=time_zone)
        end_time = handlers.wallet.convert_str_date_to_ms(date=end_time, time_zone=time_zone)
        self.start_time = start_time
        self.end_time = end_time

        # work with open timestamps
        if start_time:
            self.start_time = handlers.time_helper.open_from_milliseconds(ms=start_time, tick_interval=self.tick_interval)

        if end_time:
            self.end_time = handlers.time_helper.open_from_milliseconds(ms=end_time, tick_interval=self.tick_interval)

        # fill missing timestamps
        self.start_time, self.end_time = handlers.time_helper.time_interval(tick_interval=self.tick_interval,
                                                                            limit=self.limit,
                                                                            start=self.start_time,
                                                                            end=self.end_time)
        # discard not closed
        now = handlers.time_helper.utc()
        current_open = handlers.time_helper.open_from_milliseconds(ms=now, tick_interval=self.tick_interval)
        if self.closed:
            if self.end_time >= current_open:
                self.end_time = current_open - 1000

        #################
        # query candles #
        #################
        if not from_csv:
            self.raw = handlers.market.get_candles_by_time_stamps(symbol=self.symbol,
                                                                  tick_interval=self.tick_interval,
                                                                  start_time=self.start_time,
                                                                  end_time=self.end_time,
                                                                  limit=self.limit,
                                                                  redis_client=self.from_redis)

            dataframe = handlers.market.parse_candles_to_dataframe(raw_response=self.raw,
                                                                   columns=self.original_candles_cols,
                                                                   time_cols=self.time_cols,
                                                                   symbol=self.symbol,
                                                                   tick_interval=self.tick_interval,
                                                                   time_zone=self.time_zone,
                                                                   time_index=self.time_index)
            self.df = dataframe

        self.timestamps = self.get_timestamps()
        self.plot_splitted_serie_couples = {}
        self.len = len(self.df)

        # exchange data
        self.info_dic = handlers.exchange.get_info_dic()
        self.order_filters = self.get_order_filters()
        self.order_types = self.get_order_types()
        self.permissions = self.get_permissions()
        self.precision = self.get_precision()

    def __repr__(self):
        return str(self.df)

    ##################
    # Show variables #
    ##################

    def df(self):
        """
        Returns candles dataframe.

        :return pd.DataFrame:
        """
        return self.df

    def trades(self):
        """
        Returns trades dataframe.

        :return pd.DataFrame:
        """
        if self.trades.empty:
            print("Empty trades, please request using: get_trades() method: Example: my_symbol.get_trades()")
        return self.trades

    def symbol(self):
        """
        Returns symbol.

        :return str:
        """
        return self.symbol

    def version(self):
        """
        Returns version on BinPan.

        :return str:
        """
        return self.version

    def fees(self):
        """
        Returns fees for symbol. API keys required.

        :return dict:
        """
        return self.version

    def tick_interval(self):
        """
        Returns tick_interval for symbol.

        :return str:
        """
        return self.tick_interval

    def set_strategy_groups(self,
                            column: str,
                            group: str,
                            strategy_groups: dict = None):
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
            self.strategy_groups = handlers.tags.tag_column_to_strategy_group(column=column,
                                                                              group=group,
                                                                              strategy_groups=strategy_groups)
        return self.strategy_groups

    def get_strategy_columns(self) -> list:
        """
        Returns column names starting with "Strategy".
        :return dict: Updated strategy groups of columns.
        """

        return [i for i in self.df.columns if i.lower().startswith('strategy')]

    def start_time(self):
        """
        Returns start_time on instance.

        :return int:
        """
        return self.start_time

    def end_time(self):
        """
        Returns end_time on instance.

        :return int:
        """
        return self.end_time

    def limit(self):
        """
        Returns limit on instance.

        :return int:
        """
        return self.limit

    def time_zone(self):
        """
        Returns time_zone for symbol.

        :return str:
        """
        return self.time_zone

    def time_index(self):
        """
        Returns if time_index.

        :return bool:
        """
        return self.time_index

    def closed(self):
        """
        Returns if not closed candles were dropped when instantiated.

        :return bool:
        """
        return self.closed

    def update_info_dic(self):
        """
        Returns exchangeInfo data when instantiated. It includes, filters, fees, and many other data for all symbols in the
        exchange.

        :return dict:
        """
        self.info_dic = handlers.exchange.get_info_dic()
        return self.info_dic

    def save_csv(self,
                 timestamped: bool = True):
        """
        Saves current csv to a csv file.

        :param bool timestamped: Adds start and end timestamps to the name.
        :return:
        """
        df_ = self.df
        if timestamped:
            start, end = self.get_timestamps()
            filename = f"{df_.index.name.replace('/', '-')} {start} {end}.csv"
        else:
            filename = f"{df_.index.name.replace('/', '-')}.csv"

        handlers.files.save_dataframe_to_csv(filename=filename,
                                             data=df_,
                                             timestamp=not timestamped)
        binpan_logger.info(f"Saved file {filename}")

    ##################
    # pandas display #
    ##################

    def set_display_columns(self, display_columns=None):
        """
        Change the number of maximum columns shown in the display of the dataframe.

        :param int display_columns: Integer

        """
        if display_columns:
            pd.set_option('display.max_columns', display_columns)
        else:
            pd.set_option('display.max_columns', self.display_columns)

    def set_display_min_rows(self, display_rows=None):
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
        Change the number of maximum rows shown in the display of the dataframe.

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

    ###########
    # methods #
    ###########
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
            new_candles = handlers.market.basic_dataframe(data=self.df, exceptions=exceptions, actions_col=actions_col)
            self.df = new_candles
            return self.df
        else:
            return handlers.market.basic_dataframe(data=self.df, exceptions=exceptions, actions_col=actions_col)

    def drop(self, columns_to_drop=[], inplace=False) -> pd.DataFrame:
        """
        It drops some columns from the dataframe. If columns list not passed, then defaults to the initial columns.

        Can be used when messing with indicators to clean the object.

        :param: list columns: A list with the columns names to drop. If not passed, it defaults to the initial columns that remain
            from when instanced. Defaults to any column but initial ones.
        :param: bool inplace: When true, it drops columns in the object. False just returns a copy without that columns and dataframe
            in the object remains.
        :return pd.DataFrame: Pandas DataFrame with columns dropped.

        """
        current_columns = self.df.columns
        if not columns_to_drop:
            columns_to_drop = []
            for col in current_columns:
                if not col in self.original_candles_cols:
                    columns_to_drop.append(col)
            # self.row_counter = 1
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
                # self.row_control = {c: unique_rows_index[self.row_control[c]] if c in self.row_control.keys() else 1 for c in conserve_columns}

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

    def insert_indicator(self,
                         source_data: pd.Series or pd.DataFrame or np.ndarray or list,
                         strategy_group: str = None,
                         row: str = None,
                         rows: list = None,
                         color: str = 'blue',
                         colors: list = None,
                         color_fills: list = None,
                         name: str = None,
                         names: list = None,
                         suffix: str = '') -> pd.DataFrame or None:
        """
        Adds indicator to dataframe. It always do inplace.

        :param pd.Series or pd.DataFrame or np.ndarray or list source_data: Source data from pandas_ta or any other. Expected named series
           or list of names, or at least a suffix. If nothing passed, name will be autogenerated.
        :param int strategy_group: Optionally can be tagged into a strategy group when inserting data.
        :param int row: When a single serie inserted, a plotting row for a single inserted object. 1 overlaps candles, other, gets own row.
        :param list rows: mandatory. Rows position for autoplot each serie. 1 is overlap, ANY OTHER INTEGER will calculate row position.
           Passing list of source data will put different row for each indicator added, ignoring same number in the list. Finally you can
           change assigned row with ``my_symbol.set_plot_row('New_column', 2)``
        :param str color: When a single serie inserted, a plotting color.
        :param list colors: Colors list for each serie indicator. Default is random colors.
        :param list color_fills: Colors to fill indicator til y-axis or False to avoid. Example for transparent green
           ``'rgba(26,150,65,0.5)'``. Default is all False.
        :param str name: When a single serie inserted, a name for a single inserted object.
        :param list names: A list for the columns when inserted.
        :param str suffix: A suffix for the new column name/s. If numpy array or nameless pandas series, suffix is the whole name.
        :return pd.DataFrame or None: Instance candles dataframe.

        """
        if type(source_data) == list:
            data_len = len(source_data)
        elif type(source_data) == pd.DataFrame:
            data_len = len(source_data.columns)
        else:
            data_len = 1

        if row and not rows:
            rows = [row]
        if not rows:
            rows = [2 for _ in range(data_len)]

        if color and not colors:
            colors = [color]
        if not colors:
            colors = [choice(plotly_colors) for _ in range(data_len)]

        if not color_fills:
            color_fills = [False for _ in range(data_len)]

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
            assert len(source_data) == len(rows)
            # if not colors:
            #     colors = [choice(plotly_colors) for _ in range(len(rows))]
            # if not color_fills:
            #     color_fills = [False for _ in range(len(rows))]

            if name and not names:
                names = [f"{name}_{i}" for i in range(data_len)]

            for element_idx, new_element in enumerate(source_data):

                assert type(new_element) in [pd.Series, np.ndarray]
                data = new_element.copy(deep=True)

                if not names:
                    try:
                        current_name = new_element.name
                    except Exception:
                        current_name = f"Indicator_{len(self.df) + element_idx}{suffix}"
                else:
                    current_name = names[element_idx]

                self.insert_indicator(source_data=data,
                                      rows=[rows[element_idx]],
                                      colors=[colors[element_idx]],
                                      color_fills=[color_fills[element_idx]],
                                      names=[current_name],
                                      suffix=suffix)
            return self.df

        else:
            msg = f"BinPan Warning: Unexpected data type {type(source_data)}, expected pd.Series, np.ndarray, pd.DataFrame or list of them."
            binpan_logger.warning(msg)
            return

        if self.is_new(source_data=data, suffix=''):  # suffix is added before this to names

            rows_tags = {row: i + self.row_counter + 1 for i, row in enumerate(sorted(list(set(rows))))}
            rows = [rows_tags[r] if r != 1 else 1 for r in rows]  # downcast rows to available except 1 (overlap)

            for i, serie in enumerate(data_series):
                column_name = str(serie.name)  # suffix is added before this to names
                current_df.loc[:, column_name] = serie
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=color_fills[i])
                self.set_plot_row(indicator_column=column_name, row_position=rows[i])

            self.row_counter = max(rows)

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
        start = self.df.iloc[0]['Open timestamp']
        end = self.df.iloc[-1]['Close timestamp']
        return start, end

    def get_dates(self) -> Tuple[str, str]:
        """
        Get the first Open timestamp and the last Close timestamp, and converts to timezoned dates.

        :return tuple(int, int): Start Open date and end close date

        """
        start, end = self.get_timestamps()
        ret_start = handlers.time_helper.convert_milliseconds_to_str(ms=start, timezoned=self.time_zone)
        ret_end = handlers.time_helper.convert_milliseconds_to_str(ms=end, timezoned=self.time_zone)
        return ret_start, ret_end

    def get_trades(self,
                   hours: int = None,
                   minutes: int = None,
                   startTime: int or str = None,
                   endTime: int or str = None,
                   time_zone: str = None):
        """
        Calls the API and creates another dataframe included in the object with the aggregated trades from API for the period of the
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
        :param str time_zone: A time zone for time index conversion.

        Example:

            .. code-block::

                                                  Aggregate tradeId     Price  Quantity  First tradeId  Last tradeId                 Date      Timestamp  Buyer was maker  Best price match
                BTCBUSD Europe/Madrid
                2022-11-20 14:11:36.763000+01:00          536009627  16524.58   0.21421      632568399     632568399  2022-11-20 14:11:36  1668949896763             True              True
                2022-11-20 14:11:36.787000+01:00          536009628  16525.04   0.02224      632568400     632568400  2022-11-20 14:11:36  1668949896787             True              True
                2022-11-20 14:11:36.794000+01:00          536009629  16525.04   0.01097      632568401     632568401  2022-11-20 14:11:36  1668949896794             True              True
                2022-11-20 14:11:36.849000+01:00          536009630  16525.27   0.05260      632568402     632568403  2022-11-20 14:11:36  1668949896849            False              True
                2022-11-20 14:11:36.849000+01:00          536009631  16525.28   0.00073      632568404     632568404  2022-11-20 14:11:36  1668949896849            False              True
                ...                                             ...       ...       ...            ...           ...                  ...            ...              ...               ...
                2022-11-20 15:10:57.928000+01:00          536083210  16556.75   0.01730      632653817     632653817  2022-11-20 15:10:57  1668953457928             True              True
                2022-11-20 15:10:57.928000+01:00          536083211  16556.74   0.00851      632653818     632653819  2022-11-20 15:10:57  1668953457928             True              True
                2022-11-20 15:10:57.950000+01:00          536083212  16558.48   0.00639      632653820     632653820  2022-11-20 15:10:57  1668953457950            False              True
                2022-11-20 15:10:57.990000+01:00          536083213  16558.48   0.01242      632653821     632653821  2022-11-20 15:10:57  1668953457990             True              True
                2022-11-20 15:10:58.020000+01:00          536083214  16558.49   0.00639      632653822     632653822  2022-11-20 15:10:58  1668953458020            False              True
                [73588 rows x 9 columns]

        :return: Pandas DataFrame

        """
        if time_zone:
            self.time_zone = time_zone

        if startTime:
            handlers.wallet.convert_str_date_to_ms(date=startTime,
                                                   time_zone=self.time_zone)
        if endTime:
            handlers.wallet.convert_str_date_to_ms(date=endTime,
                                                   time_zone=self.time_zone)
        if hours:
            startTime = int(time()*1000) - (1000 * 60 * 60 * hours)
        elif minutes:
            startTime = int(time()*1000) - (1000 * 60 * minutes)

        if startTime:
            curr_startTime = startTime
        else:
            curr_startTime = self.start_time

        if endTime:
            curr_endTime = endTime
        elif self.end_time:
            curr_endTime = self.end_time
        else:
            curr_endTime = int(time()*1000)

        self.raw_trades = handlers.market.get_historical_aggregated_trades(symbol=self.symbol,
                                                                           startTime=curr_startTime,
                                                                           endTime=curr_endTime,
                                                                           redis_client_trades=self.from_redis_trades)

        self.trades = handlers.market.parse_agg_trades_to_dataframe(response=self.raw_trades,
                                                                    columns=self.trades_columns,
                                                                    symbol=self.symbol,
                                                                    time_zone=self.time_zone,
                                                                    time_index=self.time_index)
        return self.trades

    def is_new(self,
               source_data: pd.Series or pd.DataFrame,
               suffix: str = '') -> bool:
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
                binpan_logger.info(f"Existing column: {gen_col} No data added to instance.")
                return False
            else:
                binpan_logger.info(f"New column: {gen_col}")

        return True

    ################
    # Plots
    ################

    def set_plot_row(self,
                     indicator_column: str = None,
                     row_position: int = None):
        """
        Internal control formatting plots. Can be used to change plot subplot row of an indicator.

        :param str indicator_column: column name
        :param row_position: reassign row_position to column name
        :return dict: columns with its assigned row_position in subplots when plotting.

        """
        if indicator_column and row_position:
            self.row_control.update({indicator_column: row_position})
        return self.row_control

    def set_plot_color(self,
                       indicator_column: str = None,
                       color: int or str = None) -> dict:
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
        :param my_axis_group: Fill mode for indicator. Color can be forced to fill to zero line with "tozeroy" or between two indicators in same
           axis group with "tonexty".
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

    def set_plot_splitted_serie_couple(self,
                                       indicator_column_up: str = None,
                                       indicator_column_down: str = None,
                                       splitted_dfs: list = None,
                                       color_up: str = 'rgba(35, 152, 33, 0.5)',
                                       color_down: str = 'rgba(245, 63, 39, 0.5)') -> dict:
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
            self.plot_splitted_serie_couples.update({indicator_column_down: [indicator_column_up,
                                                                             indicator_column_down,
                                                                             splitted_dfs,
                                                                             color_up,
                                                                             color_down]})
        return self.plot_splitted_serie_couples

    def plot(self,
             width: int = 1800,
             height: int = 1000,
             candles_ta_height_ratio: float = 0.75,
             volume: bool = True,
             title: str = None,
             yaxis_title: str = 'Price',
             overlapped_indicators: list = [],
             priced_actions_col: str = 'Close',
             actions_col: str = None,
             marker_labels: dict = None,
             markers: list = None,
             marker_colors: list = None,
             background_color=None,
             zoom_start_idx=None,
             zoom_end_idx=None):
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

        """
        temp_df = self.df.iloc[zoom_start_idx:zoom_end_idx]

        if not title:
            title = temp_df.index.name

        indicators_series = [temp_df[k] for k in self.row_control.keys()]
        indicator_names = [temp_df[k].name for k in self.row_control.keys()]
        indicators_colors = [self.color_control[k] for k in self.row_control.keys()]
        indicators_colors = [c if type(c) == str else plotly_colors[c] for c in indicators_colors]

        rows_pos = [self.row_control[k] for k in self.row_control.keys()]

        # if actions_col:
        # if not marker_labels:
        #     marker_labels = {'buy': 1, 'sell': -1}
        # if not markers:
        #     markers = ['arrow-bar-up', 'arrow-bar-down']
        # if not marker_colors:
        #     my_marker_colors = ['red', 'green']
        #     {mark: my_marker_colors[idx % 2] for idx, mark in enumerate(marker_labels.keys())}

        if zoom_start_idx is not None or zoom_end_idx is not None:
            zoomed_plot_splitted_serie_couples = handlers.indicators.zoom_cloud_indicators(self.plot_splitted_serie_couples,
                                                                                           main_index=list(self.df.index),
                                                                                           start_idx=zoom_start_idx,
                                                                                           end_idx=zoom_end_idx)
        else:
            zoomed_plot_splitted_serie_couples = self.plot_splitted_serie_couples

        handlers.plotting.candles_tagged(data=temp_df,
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
                                         marker_colors=marker_colors)

    def plot_trades_size(self,
                         max_size: int = 60,
                         height: int = 1000,
                         logarithmic: bool = False,
                         group_big_data: int = None,
                         title: str = None):
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
        :param int group_big_data: If true, groups data in height bins, this can get faster plotting for big quantity of trades.
        :param title: Graph title

        """
        if self.trades.empty:
            binpan_logger.info("Trades not downloaded. Please add trades data with: my_symbol.get_trades()")
            return
        if not title:
            title = f"Size trade categories {self.symbol}"
        managed_data = self.trades.copy(deep=True)
        if not group_big_data:
            handlers.plotting.plot_trade_size(data=managed_data,
                                              max_size=max_size,
                                              height=height,
                                              logarithmic=logarithmic,
                                              title=title)
        else:
            # TODO: GROUP SLOTS OF TRADES
            handlers.plotting.plot_trade_size(data=managed_data,
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
            binpan_logger.info("Trades not downloaded. Please add trades data with: my_symbol.get_trades()")
            return
        if not title:
            title = f"Size trade categories {self.symbol}"
        handlers.plotting.plot_pie(serie=self.trades['Quantity'],
                                   categories=categories,
                                   logarithmic=logarithmic,
                                   title=title)

    def plot_aggression_sizes(self,
                              bins=50,
                              hist_funct='sum',
                              height=900,
                              from_trades=False,
                              title: str = None,
                              total_volume_column: str = None,
                              partial_vol_column: str = None,
                              **kwargs_update_layout):
        """
        Binance fees can be cheaper for maker orders, many times when big traders, like whales, are operating . Showing what are doing
        makers.

        It shows which kind of volume or trades came from, aggressive_sellers or aggressive_byers.

        Can be useful finding support and resistance zones.

        .. image:: images/makers_vs_takers_plot.png
           :width: 1000
           :alt: Candles with some indicators


        :param bins: How many bars.
        :param hist_funct: The way graph data is showed. It can be 'mean', 'sum', 'percent', 'probability', 'density', or 'probability density'
        :param height: Height of the graph.
        :param from_trades: Requieres grabbing trades before.
        :param title: A title.
        :param total_volume_column: The column with the total volume. It defaults automatically.
        :param partial_vol_column: The column with the partial volume. It defaults automatically. API shows maker or taker separated volumes.
        :param kwargs_update_layout: Optional

        """
        if from_trades or not self.trades.empty:
            if self.trades.empty:
                binpan_logger.info("Trades not downloaded. Please add trades data with: my_symbol.get_trades()")
                return
            else:
                _df = self.trades.copy(deep=True)

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

        handlers.plotting.plot_hists_vs(x0=aggressive_sellers,
                                        x1=aggressive_byers,
                                        x0_name="Aggressive sellers",
                                        x1_name='Aggressive byers',
                                        bins=bins,
                                        hist_funct=hist_funct,
                                        height=height,
                                        title=title,
                                        **kwargs_update_layout)

    def plot_market_profile(self,
                            bins: int = 100,
                            hours: int = None,
                            minutes: int = None,
                            startTime: int or str = None,
                            endTime: int or str = None,
                            height=900,
                            from_trades=False,
                            title: str = 'Market Profile',
                            time_zone: str = None,
                            **kwargs_update_layout):

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
        :param from_trades: Requieres grabbing trades before.
        :param title: A title.
        :param str time_zone: A time zone for time index conversion.
        :param kwargs_update_layout: Optional

        """

        if time_zone:
            self.time_zone = time_zone
        if startTime:
            handlers.wallet.convert_str_date_to_ms(date=startTime,
                                                   time_zone=self.time_zone)
        if endTime:
            handlers.wallet.convert_str_date_to_ms(date=endTime,
                                                   time_zone=self.time_zone)
        if hours:
            startTime = int(time() * 1000) - (1000 * 60 * 60 * hours)
        elif minutes:
            startTime = int(time() * 1000) - (1000 * 60 * minutes)

        if from_trades:
            if self.trades.empty:
                binpan_logger.info("Trades not downloaded. Please add trades data with: my_symbol.get_trades()")
                return
        if from_trades or not self.trades.empty:
            _df = self.trades.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            handlers.plotting.bar_plot(df=_df,
                                       x_col_to_bars='Price',
                                       y_col='Quantity',
                                       bar_segments='Buyer was maker',
                                       bins=bins,
                                       title=title,
                                       height=height,
                                       y_axis_title='Aggressive Sells VS Aggressive Buys',
                                       legend_names={'agg_Quantity_Buyer_was_maker': 'Aggressive Sell',
                                                     'agg_Quantity_not_Buyer_was_maker': 'Aggressive Buy'},
                                       **kwargs_update_layout)
        else:
            _df = self.df.copy(deep=True)
            if startTime:
                _df = _df[_df['Timestamp'] >= startTime]
            if endTime:
                _df = _df[_df['Timestamp'] <= endTime]
            binpan_logger.info(f"Using klines data. For deeper info add trades data with my_symbol.get_trades() method.")
            handlers.plotting.bar_plot(df=_df,
                                       x_col_to_bars='Close',
                                       y_col='Volume',
                                       y_axis_title='Volume levels',
                                       bins=bins,
                                       title=title,
                                       height=height,
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
            binpan_logger.info("Trades not downloaded. Please add trades data with: my_symbol.get_trades()")
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

    def plot_orderbook(self,
                       accumulated=True,
                       title='Depth orderbook plot',
                       height=800,
                       plot_y="Quantity",
                       **kwargs):
        """
        Plots orderbook depth.
        """
        if self.orderbook.empty:
            binpan_logger.info("Orderbook not downloaded. Please add orderbook data with: my_binpan.get_orderbook()")
            return
        handlers.plotting.orderbook_depth(df=self.orderbook,
                                          accumulated=accumulated,
                                          title=title,
                                          height=height,
                                          plot_y=plot_y,
                                          **kwargs)

    def plot_orderbook_density(self,
                               x_col="Price",
                               color='Side',
                               bins=300,
                               histnorm: str = 'density',
                               height: int = 800,
                               title: str = None,
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

        handlers.plotting.dist_plot(df=self.orderbook,
                                    x_col=x_col,
                                    color=color,
                                    bins=bins,
                                    histnorm=histnorm,
                                    height=height,
                                    title=title,
                                    **update_layout_kwargs)

    #################
    # Exchange data #
    #################

    def get_order_filters(self) -> dict:
        """
        Get exchange info about the symbol for order filters.
        :return dict:
        """
        filters = handlers.exchange.get_symbols_filters(info_dic=self.info_dic)
        self.order_filters = filters[self.symbol]
        return self.order_filters

    def get_order_types(self) -> dict:
        """
        Get exchange info about the symbol for order types.
        :return dict:
        """
        order_types_precision = handlers.exchange.get_orderTypes_and_permissions(info_dic=self.info_dic)
        self.order_types = order_types_precision[self.symbol]['orderTypes']
        return self.order_types

    def get_permissions(self) -> dict:
        """
        Get exchange info about the symbol for trading permissions.
        :return dict:
        """
        permissions = handlers.exchange.get_orderTypes_and_permissions(info_dic=self.info_dic)
        self.permissions = permissions[self.symbol]['permissions']
        return self.permissions

    def get_precision(self) -> dict:
        """
        Get exchange info about the symbol for assets precision.
        :return dict:
        """
        precision = handlers.exchange.get_precision(info_dic=self.info_dic)
        self.precision = precision[self.symbol]
        return self.precision

    def get_status(self) -> str:
        """
        Return the symbol status, TRADING, BREAK, etc.
        """
        return Exchange().df.loc[self.symbol].to_dict()['status']

    ###############
    # Backtesting #
    ###############

    # def set_action_labels(self,
    #                       indicator_column: str = None,
    #                       label_in: str = 'buy',
    #                       label_out: str = 'sell') -> dict:
    #     """
    #     Sets labels to use when backtesting over a column of actions. In means you obtain base, out means you obtain quote.
    #
    #     DEPRECATED, ACTION LABELS MUST BE -1 OR 1.
    #
    #     :param str indicator_column: A column name of a BinPan dataframe colum.
    #     :param str label_in: A label to register a sell point.
    #     :param str label_out: A label to register a buy point.
    #     :return dict: Dictionary with action columns labels.
    #     """
    #     if indicator_column and label_in and label_out:
    #         self.action_labels.update({indicator_column: {'label_in': label_in, 'label_out': label_out}})
    #     return self.action_labels

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
                    label_in=1,
                    label_out=-1,
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

        # if self.symbol in self.action_labels.keys():
        #     label_in = self.action_labels[self.symbol]['label_in']
        #     label_out = self.action_labels[self.symbol]['label_out']

        if type(actions_col) == int:
            actions = self.df.iloc[:, actions_col]
        else:
            actions = self.df[actions_col]

        if suffix:
            suffix = '_' + suffix

        if not short:
            wallet_df = handlers.tags.backtesting(df=self.df,
                                                  actions_column=actions,
                                                  target_column=target_column,
                                                  stop_loss_column=stop_loss_column,
                                                  entry_filter_column=entry_filter_column,
                                                  priced_actions_col=priced_actions_col,
                                                  fixed_target=fixed_target,
                                                  fixed_stop_loss=fixed_stop_loss,
                                                  base=base,
                                                  quote=quote,
                                                  fee=fee,
                                                  label_in=label_in,
                                                  label_out=label_out,
                                                  suffix=suffix,
                                                  evaluating_quote=evaluating_quote,
                                                  info_dic=self.info_dic)
        else:
            wallet_df = handlers.tags.backtesting_short(df=self.df,
                                                        actions_column=actions,
                                                        target_column=target_column,
                                                        stop_loss_column=stop_loss_column,
                                                        entry_filter_column=entry_filter_column,
                                                        priced_actions_col=priced_actions_col,
                                                        fixed_target=fixed_target,
                                                        fixed_stop_loss=fixed_stop_loss,
                                                        base=base,
                                                        quote=quote,
                                                        fee=fee,
                                                        label_in=label_in,
                                                        label_out=label_out,
                                                        suffix=suffix,
                                                        evaluating_quote=evaluating_quote,
                                                        info_dic=self.info_dic)

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

    def roi(self,
            column: str = None) -> float:
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

    def profit_hour(self,
                    column: str = None) -> float:
        """
        It returns win or loos quantity per hour. Just compares first and last value. Expected datetime index. If not column passed, it
        will search for an Evaluation column.

        :param str column: A column in the BinPan's DataFrame with values to check profit with expected datetime index.
        :return float: Resulting return of inversion.
        """
        if not column:
            column = [i for i in self.df.columns if i.startswith('Eval')][-1]
            print(f"Auto selected column {column}")

        my_column = self.df[column].copy(deep=True)
        my_column.dropna(inplace=True)

        first = my_column.iloc[0]
        last = my_column.iloc[-1]

        profit = last - first
        ms = self.df['Close timestamp'].dropna().iloc[-1] - self.df['Open timestamp'].dropna().iloc[0]
        hours = ms / (1000 * 60 * 60)

        print(f"Total profit for {column}: {profit}")

        return profit / hours

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
            return handlers.exchange.get_fees(symbol=symbol, decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)

        except Exception as _:
            binpan_logger.warning("Fees cannot be requested without api key added. Add it with"
                                  " binpan.handlers.files_filters.add_api_key('xxxxxxxxxx')")

    def get_orderbook(self,
                      limit: int = 5000) -> pd.DataFrame:
        """
        Gets orderbook.

        :param int limit: Defaults to maximum: 5000
        :return pd.DataFrame:
        """
        orders = handlers.market.get_order_book(symbol=self.symbol,
                                                limit=limit)
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

        if 'length' in kwargs.keys():
            if kwargs['length'] >= len(self.df):
                msg = f"BinPan Error: Ma window larger than data length."
                binpan_logger.error(msg)
                raise Exception(msg)
        if suffix:
            kwargs.update({'suffix': suffix})

        df = self.df.copy(deep=True)
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
        return self.ma(ma_name='ema', column_source=column, inplace=inplace, length=window, suffix=suffix, color=color, **kwargs)

    def supertrend(self,
                   length: int = 10,
                   multiplier: int = 3,
                   inplace=True,
                   suffix: str = None,
                   colors: list = None,
                   **kwargs):
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
        supertrend_df = ta.supertrend(high=self.df['High'],
                                      low=self.df['Low'],
                                      close=self.df['Close'],
                                      length=length,
                                      multiplier=int(multiplier),
                                      **kwargs)
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

    def macd(self, fast: int = 12,
             slow: int = 26,
             smooth: int = 9,
             inplace: bool = True,
             suffix: str = '',
             colors: list = ['black', 'orange', 'green', 'blue'],
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

        .. image:: images/indicators/macd.png
           :width: 1000
           :alt: Candles with some indicators

        """
        macd = self.df.ta.macd(fast=fast,
                               slow=slow,
                               signal=smooth,
                               **kwargs)
        zeros = macd.iloc[:, 0].copy()
        zeros.loc[:] = 0
        zeros.name = 'zeros'
        macd = pd.concat([zeros, macd], axis=1, ignore_index=False)

        if inplace and self.is_new(macd):
            self.row_counter += 1

            self.global_axis_group -= 1
            axis_identifier = f"y{self.global_axis_group}"

            for i, column_name in enumerate(macd.columns):
                col = macd[column_name]
                column_name = str(col.name) + suffix
                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_row(indicator_column=column_name, row_position=self.row_counter)
                self.set_plot_axis_group(indicator_column=column_name, my_axis_group=axis_identifier)

                if column_name.startswith('MACDh_'):

                    splitted_dfs = handlers.indicators.df_splitter(data=macd,
                                                                   up_column=column_name,
                                                                   down_column='zeros')

                    self.set_plot_splitted_serie_couple(indicator_column_up=column_name,
                                                        indicator_column_down='zeros',
                                                        splitted_dfs=splitted_dfs,
                                                        color_up='rgba(35, 152, 33, 0.5)',
                                                        color_down='rgba(245, 63, 39, 0.5)')

                else:
                    self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
                    self.set_plot_filled_mode(indicator_column=column_name, fill_mode=None)

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

        .. image:: images/indicators/rsi.png
           :width: 1000
           :alt: Candles with some indicators

        """

        rsi = ta.rsi(close=self.df['Close'],
                     length=length,
                     **kwargs)
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

        .. image:: images/indicators/stochrsi.png
           :width: 1000
           :alt: Candles with some indicators

        """
        stoch_df = ta.stochrsi(close=self.df['Close'],
                               length=rsi_length,
                               rsi_length=rsi_length,
                               k_smooth=k_smooth,
                               d_smooth=d_smooth,
                               **kwargs)
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

        .. image:: images/indicators/on_balance.png
           :width: 1000
           :alt: Candles with some indicators

        """

        on_balance = ta.obv(close=self.df['Close'],
                            volume=self.df['Volume'],
                            **kwargs)

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

        .. image:: images/indicators/ad.png
           :width: 1000
           :alt: Candles with some indicators

        """

        ad = ta.ad(high=self.df['High'],
                   low=self.df['Low'],
                   close=self.df['Close'],
                   volume=self.df['Volume'],
                   **kwargs)

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

        .. image:: images/indicators/vwap.png
           :width: 1000
           :alt: Candles with some indicators

        """

        vwap = ta.vwap(high=self.df['High'],
                       low=self.df['Low'],
                       close=self.df['Close'],
                       volume=self.df['Volume'],
                       anchor=anchor,
                       **kwargs)

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

        .. image:: images/indicators/atr.png
           :width: 1000
           :alt: Candles with some indicators

        """

        atr = ta.atr(high=self.df['High'],
                     low=self.df['Low'],
                     close=self.df['Close'],
                     length=length,
                     **kwargs)

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

        .. image:: images/indicators/cci.png
           :width: 1000
           :alt: Candles with some indicators

        """

        cci = ta.cci(high=self.df['High'],
                     low=self.df['Low'],
                     close=self.df['Close'],
                     length=length,
                     c=scaling,
                     **kwargs)

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

    def eom(self,
            length: int = 14,
            divisor: int = 100000000,
            drift: int = 1,
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

        .. image:: images/indicators/eom.png
           :width: 1000
           :alt: Candles with some indicators

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

        if inplace and self.is_new(eom):
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

        .. image:: images/indicators/roc.png
           :width: 1000
           :alt: Candles with some indicators

        """

        roc = ta.roc(close=self.df['Close'],
                     length=length,
                     escalar=escalar,
                     **kwargs)

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

    def bbands(self,
               length: int = 5,
               std: int = 2,
               ddof: int = 0,
               inplace: bool = True,
               suffix: str = '',
               colors: list = ['red', 'orange', 'green'],
               my_fill_color: str = 'rgba(47, 48, 56, 0.2)',
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
        :param str my_fill_color: An rgba color code to fill between bands area. https://rgbacolorpicker.com/
        :param kwargs: Optional from https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volatility/bbands.py
        :return: pd.Series

        .. image:: images/indicators/bbands.png
           :width: 1000
           :alt: Candles with some indicators

        """
        bbands = self.df.ta.bbands(close=self.df['Close'],
                                   length=length,
                                   std=std,
                                   ddof=ddof,
                                   suffix=suffix,
                                   **kwargs)

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

    def stoch(self,
              k_length: int = 14,
              stoch_d=3,
              k_smooth: int = 1,
              inplace: bool = True,
              suffix: str = '',
              colors: list = ['orange', 'blue'],
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
        stoch_df = ta.stoch(high=self.df['High'],
                            low=self.df['Low'],
                            close=self.df['Close'],
                            k=k_length,
                            d=stoch_d,
                            k_smooth=k_smooth,
                            **kwargs)
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

    def ichimoku(self,
                 tenkan: int = 9,
                 kijun: int = 26,
                 chikou_span: int = 26,
                 senkou_cloud_base: int = 52,
                 inplace: bool = True,
                 suffix: str = '',
                 colors: list = ['orange', 'skyblue', 'grey', 'green', 'red']):
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

        ichimoku_data = handlers.indicators.ichimoku(data=self.df,
                                                     tenkan=tenkan,
                                                     kijun=kijun,
                                                     chikou_span=chikou_span,
                                                     senkou_cloud_base=senkou_cloud_base,
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

                    splitted_dfs = handlers.indicators.df_splitter(data=ichimoku_data,
                                                                   up_column=pre_col_name,
                                                                   down_column=column_name)

                    self.set_plot_splitted_serie_couple(indicator_column_up=pre_col_name,
                                                        indicator_column_down=column_name,
                                                        splitted_dfs=splitted_dfs,
                                                        color_up='rgba(35, 152, 33, 0.5)',
                                                        color_down='rgba(245, 63, 39, 0.5)')

        return ichimoku_data

    def fractal_w(self,
                  period: int = 2,
                  inplace: bool = True,
                  suffix: str = '',
                  colors: list = ['orange', 'skyblue']):
        """
        The fractal indicator is based on a simple price pattern that is frequently seen in financial markets. Outside of trading, a fractal
        is a recurring geometric pattern that is repeated on all time frames. From this concept, the fractal indicator was devised.
        The indicator isolates potential turning points on a price chart. It then draws arrows to indicate the existence of a pattern.

        https://www.investopedia.com/terms/f/fractal.asp

        From: https://codereview.stackexchange.com/questions/259703/william-fractal-technical-indicator-implementation

        :param int period: Default is 2. Count of neighbour candles to match max or min tags.
        :return pd.Series: A serie with 1 or -1 for local max or local min to tag.
        :param bool inplace: Make it permanent in the instance or not.
        :param str suffix: A decorative suffix for the name of the column created.
        :param list colors: A list of colors for the indicator dataframe columns. Is the color to show when plotting.
            It can be any color from plotly library or a number in the list of those. Default colors defined.
            https://community.plotly.com/t/plotly-colours-list/11730

        """

        fractal = handlers.indicators.fractal_w(data=self.df, period=period, suffix=suffix)

        if inplace and self.is_new(fractal):

            binpan_logger.debug(fractal.columns)

            for i, column_name in enumerate(fractal.columns):
                self.row_counter += 1

                col_data = fractal[column_name]

                self.df.loc[:, column_name] = col_data

                self.set_plot_color(indicator_column=column_name, color=colors[i])
                self.set_plot_color_fill(indicator_column=column_name, color_fill=False)
                self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)

        return fractal

    @staticmethod
    def pandas_ta_indicator(name: str,
                            **kwargs):
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
        # TODO: add to autoplot
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
            inplace=True,
            suffix: str = '',
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

           from binpan import binpan

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

        compared = handlers.tags.tag_comparison(serie_a=data_a,
                                                serie_b=data_b,
                                                **{relation: True},
                                                match_tag=match_tag,
                                                mismatch_tag=mismatch_tag)

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
            self.strategy_groups = handlers.tags.tag_column_to_strategy_group(column=column_name,
                                                                              group=strategy_group,
                                                                              strategy_groups=self.strategy_groups)
        return compared

    def cross(self,
              slow: str or int or float or pd.Series,
              fast: str or int or pd.Series = 'Close',
              cross_over_tag: str or int = 1,
              cross_below_tag: str or int = -1,
              echo=0,
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

           from binpan import binpan

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

        cross = handlers.tags.tag_cross(serie_a=data_a,
                                        serie_b=data_b,
                                        echo=echo,
                                        cross_over_tag=cross_over_tag,
                                        cross_below_tag=cross_below_tag,
                                        name=column_name,
                                        non_zeros=non_zeros)

        if inplace and self.is_new(cross):
            self.row_counter += 1
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)  # overlaps are one
            self.df.loc[:, column_name] = cross

        if strategy_group:
            self.strategy_groups = handlers.tags.tag_column_to_strategy_group(column=column_name,
                                                                              group=strategy_group,
                                                                              strategy_groups=self.strategy_groups)
        return cross

    def shift(self,
              column: str or int or pd.Series,
              window=1,
              strategy_group: str = '',
              inplace=True,
              suffix: str = '',
              color: str or int = 'grey'
              ):
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
        shift = handlers.indicators.shift_indicator(serie=data_a, window=window)
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
            self.strategy_groups = handlers.tags.tag_column_to_strategy_group(column=column_name,
                                                                              group=strategy_group,
                                                                              strategy_groups=self.strategy_groups)

        return shift

    def merge_columns(self,
                      main_column: str or int or pd.Series,
                      other_column: str or int or pd.Series,
                      sign_other: dict = {1: -1},
                      strategy_group: str = '',
                      inplace=True,
                      suffix: str = '',
                      color: str or int = 'grey'
                      ):
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

        merged = handlers.tags.merge_series(predominant=data_a,
                                            other=data_b)
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
            self.strategy_groups = handlers.tags.tag_column_to_strategy_group(column=column_name,
                                                                              group=strategy_group,
                                                                              strategy_groups=self.strategy_groups)
        return merged

    def clean_in_out(self,
                     column: str or int or pd.Series,
                     in_tag=1,
                     out_tag=-1,
                     strategy_group: str = '',
                     inplace=True,
                     suffix: str = '',
                     color: str or int = 'grey'):
        """
        Predominant serie will be filled nans with values, if existing, from the other serie.

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
            data_a = self.df.iloc[:, column]
        else:
            data_a = column.copy(deep=True)

        # if data_a.name in self.action_labels.keys():
        #     in_tag = self.action_labels[data_a.name]['label_in']
        #     out_tag = self.action_labels[data_a.name]['label_out']

        clean = handlers.tags.clean_in_out(serie=data_a,
                                           in_tag=in_tag,
                                           out_tag=out_tag)
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
            self.strategy_groups = handlers.tags.tag_column_to_strategy_group(column=column_name,
                                                                              group=strategy_group,
                                                                              strategy_groups=self.strategy_groups)
        return clean

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

    def ffill(self,
              column: str or int or pd.Series,
              window: int = 1,
              inplace=True,
              replace=False,
              suffix: str = '',
              color: str or int = 'blue'):
        """
        It forward fills a value through nans through a window ahead.

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

        my_ffill = handlers.indicators.ffill_indicator(serie=serie, window=window)

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


class Exchange(object):
    """
    Exchange data.

    Exchange data collected in class variables:

    - **my_exchange_instance.info_dic**: A dictionary with all raw symbols info each.
    - **my_exchange_instance.coins_dic**: A dictionary with all coins info.
    - **my_exchange_instance.bases**: A dictionary with all bases for all symbols.
    - **my_exchange_instance.quotes**: A dictionary with all quotes for all symbols.
    - **my_exchange_instance.leveraged**: A list with all leveraged coins.
    - **my_exchange_instance.leveraged_symbols**: A list with all leveraged symbols.
    - **my_exchange_instance.fees**: dataframe with fees applied to the user requesting for every symbol.
    - **my_exchange_instance.filters**: A dictionary with all trading filters detailed with all symbols.
    - **my_exchange_instance.status**: API status can be normal o under maintenance.
    - **my_exchange_instance.coins**: A dataframe with all the coin's data.
    - **my_exchange_instance.networks**: A dataframe with info about every coin and its blockchain networks info.
    - **my_exchange_instance.coins_list**: A list with all the coin's names.
    - **my_exchange_instance.symbols**: A list with all the symbols names.
    - **my_exchange_instance.df**: A dataframe with all the symbols info.
    - **my_exchange_instance.order_types**: Dataframe with each symbol order types.

    """

    def __init__(self):
        try:
            secret_module = importlib.import_module('secret')
            importlib.reload(secret_module)
            self.api_key = secret_module.api_key
            self.api_secret = secret_module.api_secret
        except ImportError:
            raise handlers.exceptions.MissingApiData(f"Binance Api key or Api Secret not found.")
        except KeyError:
            raise handlers.exceptions.MissingApiData(f"Binance Api key or Api Secret not found.")

        self.info_dic = handlers.exchange.get_info_dic()
        self.coins_dic = handlers.exchange.get_coins_info_dic(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.bases = handlers.exchange.get_bases_dic(info_dic=self.info_dic)
        self.quotes = handlers.exchange.get_quotes_dic(info_dic=self.info_dic)
        self.leveraged = handlers.exchange.get_leveraged_coins(coins_dic=self.coins_dic, decimal_mode=False, api_key=self.api_key,
                                                               api_secret=self.api_secret)
        self.leveraged_symbols = handlers.exchange.get_leveraged_symbols(info_dic=self.info_dic, leveraged_coins=self.leveraged,
                                                                         decimal_mode=False, api_key=self.api_key,
                                                                         api_secret=self.api_secret)

        self.fees = handlers.exchange.get_fees(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.filters = handlers.exchange.get_symbols_filters(info_dic=self.info_dic)
        self.status = handlers.exchange.get_system_status()
        self.coins, self.networks = handlers.exchange.get_coins_and_networks_info(decimal_mode=False, api_key=self.api_key,
                                                                                  api_secret=self.api_secret)
        self.coins_list = list(self.coins.index)

        self.symbols = self.get_symbols()
        self.df = self.get_df()
        self.order_types = self.get_order_types()

    def __repr__(self):
        return str(self.df)

    def filter(self, symbol: str):
        """
        Returns exchange filters applied for orders with the selected symbol.

        :param str symbol:
        :return dict:
        """
        return self.filters[symbol.upper()]

    def fee(self, symbol: str):
        """
        Returns exchange fees applied for orders with the selected symbol.

        :param str symbol:
        :return pd.Series:
        """
        return self.fees.loc[symbol.upper()]

    def coin(self, coin: str):
        """
        Returns coin exchange info in a pandas serie.

        :param str coin:
        :return pd.Series:
        """
        return self.coins.loc[coin.upper()]

    def network(self, coin: str):
        """
        Returns a dataframe with all exchange networks of one specified coin or every coin.

        :param str coin:
        :return pd.Series:
        """
        return self.networks.loc[coin.upper()]

    def update_info(self, symbol: str = None):
        """
        Updates from API and returns a dict with all merged exchange info about a symbol.

        :param str symbol:
        :return dict:
        """
        self.info_dic = handlers.exchange.get_info_dic()
        self.filters = handlers.exchange.get_symbols_filters(info_dic=self.info_dic)
        self.bases = handlers.exchange.get_bases_dic(info_dic=self.info_dic)
        self.quotes = handlers.exchange.get_quotes_dic(info_dic=self.info_dic)
        self.fees = handlers.exchange.get_fees(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.status = handlers.exchange.get_system_status()
        self.coins, self.networks = handlers.exchange.get_coins_and_networks_info(decimal_mode=False, api_key=self.api_key,
                                                                                  api_secret=self.api_secret)
        self.symbols = self.get_symbols()
        self.df = self.get_df()
        self.order_types = self.get_order_types()

        if symbol:
            return self.info_dic[symbol.upper()]
        return self.info_dic

    def get_symbols(self, coin: str = None, base: bool = True, quote: bool = True):
        """
        Return list of symbols for a coin. Can be selected symbols where it is base, or quote, or both.

        :param str coin: An existing binance coin.
        :param bool base: Activate return of symbols where coin is base.
        :param bool quote: Activate return of symbols where coin is quote.
        :return list: List of symbols where it is base, quote or both.
        """
        if not coin:
            self.symbols = list(self.info_dic.keys())
            return self.symbols

        else:
            ret = []
            if base:
                ret += [i for i in self.symbols if self.bases[i] == coin.upper()]
            if quote:
                ret += [i for i in self.symbols if self.quotes[i] == coin.upper()]
            return ret

    def get_df(self):
        """
        Extended symbols dataframe with exchange info about trading permissions, trading or blocked symbol, order types, margin allowed, etc

        :return pd.DataFrame: An exchange dataframe with all symbols data.

        """
        df = pd.DataFrame(self.info_dic.values()).set_index('symbol', drop=True)
        per_df = pd.DataFrame(data=df['permissions'].tolist(), index=df.index)
        per_df.columns = per_df.value_counts().index[0]
        self.df = pd.concat([df.drop('permissions', axis=1), per_df.astype(bool)], axis=1)
        return self.df

    def get_order_types(self):
        """
        Returns a dataframe with order types for symbol.

        :return pd.DataFrame:
        """
        ord_df = pd.DataFrame(data=self.df['orderTypes'].tolist(), index=self.df.index)
        ord_df.columns = ord_df.value_counts().index[0]
        self.order_types = ord_df.astype(bool)
        return self.order_types


class Wallet(object):
    """
    Wallet is a BinPan Class that can give information about balances, in Spot or Margin trading.

    Also can show snapshots of the account status days ago, or using timestamps.

    """

    def __init__(self,
                 time_zone='UTC',
                 snapshot_days: int = 30):
        try:
            secret_module = importlib.import_module('secret')
            importlib.reload(secret_module)
            self.api_key = secret_module.api_key
            self.api_secret = secret_module.api_secret
        except ImportError:
            raise handlers.exceptions.MissingApiData(f"Binance Api key or Api Secret not found.")
        except KeyError:
            raise handlers.exceptions.MissingApiData(f"Binance Api key or Api Secret not found.")

        self.time_zone = time_zone
        self.spot = self.update_spot()
        self.spot_snapshot = None
        self.spot_startTime = None
        self.spot_endTime = None
        self.spot_requested_days = snapshot_days

        self.margin = self.update_margin()
        self.margin_snapshot = None
        self.margin_startTime = None
        self.margin_endTime = None
        self.margin_requested_days = snapshot_days

    def update_spot(self, decimal_mode=False):
        """
        Updates balances in the class object.
        :param bool decimal_mode: Use of decimal objects instead of float.
        :return dict: Wallet dictionary
        """
        self.spot = handlers.wallet.get_spot_balances_df(decimal_mode=decimal_mode, api_key=self.api_key, api_secret=self.api_secret)
        return self.spot

    def spot_snapshot(self,
                      startTime: int or str = None,
                      endTime: int or str = None,
                      snapshot_days=30,
                      time_zone=None):
        """
        Updates spot wallet snapshot.

        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int snapshot_days: Days to look if not start time or endtime passed.
        :param str time_zone: A time zone for time index conversion.
        :return pd.DataFrame: Spot wallet snapshot for the time period requested.
        """
        if time_zone:
            self.time_zone = time_zone

        self.spot_startTime = startTime
        self.spot_snapshot = handlers.wallet.daily_account_snapshot(account_type='SPOT',
                                                                    startTime=handlers.wallet.convert_str_date_to_ms(date=startTime,
                                                                                                                     time_zone=self.time_zone),
                                                                    endTime=handlers.wallet.convert_str_date_to_ms(date=endTime,
                                                                                                                   time_zone=self.time_zone),
                                                                    limit=snapshot_days,
                                                                    time_zone=self.time_zone,
                                                                    decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.spot_endTime = endTime
        self.spot_requested_days = snapshot_days

        return self.spot

    def update_margin(self, decimal_mode=False):
        """
        Updates balances in the wallet class object.

        :param bool decimal_mode: Use of decimal objects instead of float.
        :return dict: Wallet dictionary
        """
        my_margin = handlers.wallet.get_margin_balances(decimal_mode=decimal_mode, api_key=self.api_key, api_secret=self.api_secret)
        self.margin = pd.DataFrame(my_margin).T
        self.margin.index.name = 'asset'
        return self.margin

    def margin_snapshot(self,
                        startTime: int or str = None,
                        endTime: int or str = None,
                        snapshot_days=30,
                        time_zone=None):
        """
        Updates margin wallet snapshot.

        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int snapshot_days: Days to look if not start time or endtime passed.
        :param str time_zone: A time zone for time index conversion.
        :return pd.DataFrame: Spot wallet snapshot for the time period requested.
        """
        if time_zone:
            self.time_zone = time_zone

        self.spot = handlers.wallet.daily_account_snapshot(account_type='MARGIN',
                                                           startTime=handlers.wallet.convert_str_date_to_ms(date=startTime,
                                                                                                            time_zone=self.time_zone),
                                                           endTime=handlers.wallet.convert_str_date_to_ms(date=endTime,
                                                                                                          time_zone=self.time_zone),
                                                           limit=snapshot_days,
                                                           time_zone=self.time_zone,
                                                           decimal_mode=False,
                                                           api_key=self.api_key,
                                                           api_secret=self.api_secret)
        self.margin_startTime = startTime
        self.margin_endTime = endTime
        self.margin_requested_days = snapshot_days
        return self.margin

    def spot_wallet_performance(self,
                                decimal_mode: bool,
                                startTime=None,
                                endTime=None,
                                days: int = 30,
                                convert_to: str = 'BUSD'):
        """
        Calculate difference between current wallet not locked values and days before.
        :param bool decimal_mode: Fixes Decimal return type and operative.
        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int days: Days to compare balances.
        :param str convert_to: Converts balances to a coin.
        :return float: Value increase or decrease with current value of convert_to coin.
        """
        if days != self.spot_requested_days or startTime != self.spot_startTime or endTime != self.spot_endTime:
            self.spot = handlers.wallet.daily_account_snapshot(account_type='SPOT',
                                                               startTime=handlers.wallet.convert_str_date_to_ms(date=startTime,
                                                                                                                time_zone=self.time_zone),
                                                               endTime=handlers.wallet.convert_str_date_to_ms(date=endTime,
                                                                                                              time_zone=self.time_zone),
                                                               limit=days,
                                                               time_zone=self.time_zone,
                                                               decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
            self.spot_startTime = startTime
            self.spot_endTime = endTime
            self.spot_requested_days = days

        if not self.spot.empty:
            totalAssetOfBtc = self.spot['totalAssetOfBtc'].tolist()
            performance = totalAssetOfBtc[-1] - totalAssetOfBtc[0]
            if convert_to == 'BTC':
                return performance
            else:
                return handlers.market.convert_coin(coin='BTC',
                                                    convert_to=convert_to,
                                                    coin_qty=performance, decimal_mode=decimal_mode)
        else:
            return 0

    def margin_wallet_performance(self,
                                  decimal_mode: bool,
                                  startTime=None,
                                  endTime=None,
                                  days: int = 30,
                                  convert_to: str = 'BUSD'):
        """
        Calculate difference between current wallet not locked values and days before.
        :param bool decimal_mode: Fixes Decimal return type and operative.
        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int days: Days to compare balances.
        :param str convert_to: Converts balances to a coin.
        :return float: Value increase or decrease with current value of convert_to coin.
        """
        if days != self.margin_requested_days or startTime != self.margin_startTime or endTime != self.margin_endTime:
            self.margin = handlers.wallet.daily_account_snapshot(account_type='MARGIN',
                                                                 startTime=handlers.wallet.convert_str_date_to_ms(date=startTime,
                                                                                                                  time_zone=self.time_zone),
                                                                 endTime=handlers.wallet.convert_str_date_to_ms(date=endTime,
                                                                                                                time_zone=self.time_zone),
                                                                 limit=days,
                                                                 time_zone=self.time_zone,
                                                                 decimal_mode=False,
                                                                 api_key=self.api_key,
                                                                 api_secret=self.api_secret)
            self.margin_startTime = startTime
            self.margin_endTime = endTime
            self.margin_requested_days = days

        if not self.margin.empty:
            totalAssetOfBtc = self.margin['totalAssetOfBtc'].tolist()
            performance = totalAssetOfBtc[-1] - totalAssetOfBtc[0]
            if convert_to == 'BTC':
                return performance
            else:
                return handlers.market.convert_coin(coin='BTC',
                                                    convert_to=convert_to,
                                                    coin_qty=performance,
                                                    decimal_mode=decimal_mode)
        else:
            return 0


def redis_client(ip: str = '127.0.0.1',
                 port: int = 6379,
                 db: int = 0,
                 decode_responses: bool = True,
                 **kwargs):
    """
    A redis consumer client creator for the Redis module.

    :param str ip: Redis host ip. Default is localhost.
    :param int port: Default is 6379
    :param int db: Default is 0.
    :param bool decode_responses: It decodes responses from redis, avoiding bytes objects to be returned. Default is True.
    :param kwargs: If passed, object is instantiated exclusively with kwargs, discarding any passed parameter.
    :return object: A redis client.
    """
    from redis import StrictRedis

    if kwargs:
        return StrictRedis(**kwargs)
    else:
        # noinspection PyTypeChecker
        return StrictRedis(host=ip,
                           port=port,
                           db=db,
                           decode_responses=decode_responses)
