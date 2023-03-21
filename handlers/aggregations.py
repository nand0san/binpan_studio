"""
Data Aggregation.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

import handlers.market
# from .time_helper import pandas_freq_tick_interval

from .exceptions import BinPanException


# from .market import atomic_trades_columns_from_redis, atomic_trades_columns_from_binance, agg_trades_columns_from_binance, agg_trades_columns_from_redis


# TODO: add documentation

################
# Aggregations #
################

def generate_count_grouper_column(data: pd.DataFrame, grouper_name: str, size: int) -> pd.DataFrame:
    """
    It adds a column with a value for each rows count.

    :param pd.Dataframe data: A dataframe with at least two columns.
    :param str grouper_name: Name for the column to be used as grouper.
    :param int size: Count of rows to declare grouper integer sequence.
    :return: A copy of the dataframe with a new column.
    """
    df = data.copy(deep=True)
    cols = list(df.columns)
    df['count'] = 1
    df.iloc[0, -1] = 0
    df['acc_count'] = df['count'].cumsum()
    df[grouper_name] = df['acc_count'] // size
    cols.append(grouper_name)
    return df[cols]


def ohlc_group(data: pd.DataFrame, column_to_ohlc: str, group_column: str) -> pd.DataFrame:
    """
    Creates  OHLC columns for a column based on group by other column with discrete values.

    :param pd.Dataframe data: A dataframe with at least two columns.
    :param str column_to_ohlc: Column to sparse values to OHLC columns in each group.
    :param str group_column: This column will be the grouping key.
    :return: A copy of the dataframe with OHLC data.
    """
    df = data.copy(deep=True)
    df['Open'] = df.groupby([group_column])[column_to_ohlc].transform('first')
    df['High'] = df.groupby([group_column])[column_to_ohlc].transform('max')
    df['Low'] = df.groupby([group_column])[column_to_ohlc].transform('min')
    df['Close'] = df.groupby([group_column])[column_to_ohlc].transform('last')
    return df


def sum_split_by_boolean_column_and_group(data: pd.DataFrame, column_to_split_sum: str, bool_col: str, group_column: str) -> pd.DataFrame:
    """
    If a boolean column in a dataframe with a grouper column, it splits data into two columns for true and false by the grouper.

    :param pd.Dataframe data: A dataframe with at least 3 columns.
    :param str column_to_split_sum: Numeric column with data to sum by group and split by boolean column.
    :param str bool_col: Column to define splitting.
    :param str group_column: This column will be the grouping key.
    :return: A copy of the dataframe with splitted sum by group from column to split sum.
    """
    grouper = [group_column, bool_col]
    df = data.copy(deep=True)
    cols = list(df.columns)
    sum_serie = df.groupby(grouper)[column_to_split_sum].transform('sum')
    df['sum'] = sum_serie
    split1 = f'{column_to_split_sum} {bool_col} True'
    split2 = f'{column_to_split_sum} {bool_col} False'
    cols.append(split1)
    cols.append(split2)
    idx1 = df[df[bool_col] == True].index
    idx2 = df[df[bool_col] == False].index
    df.loc[idx1, split1] = sum_serie.loc[idx1]
    df.loc[idx2, split2] = sum_serie.loc[idx2]
    # df.fillna(inplace=True, method='ffill')
    # df.fillna(inplace=True, method='bfill')
    return df[cols]


def drop_aggregated(data: pd.DataFrame, group_column: str, by='last') -> pd.DataFrame:
    """
    Drop lines except the first/last/min/max etc row of each group_column streak.

    :param pd.Dataframe data: A dataframe.
    :param str group_column: This column will be the grouping key.
    :param str by: Pandas valid aggregation method.
    :return: A copy of the dataframe with just the first row each grouper streak.
    """
    df = data.copy(deep=True)
    aggregator = {c: by for c in df.columns}
    return df.groupby(group_column).agg(aggregator)


def tag_by_accumulation(trades: pd.DataFrame, threshold: int, agg_column: str = 'Quantity', grouper_name: str = 'group') -> pd.DataFrame:
    """
    Creates integer sequence by column value threshold accumulation.

    :param pd.DataFrame trades: Expected binpan aggregated trades or atomic trades dataframe.
    :param str agg_column: Name of the column to group by volume accumulation.
    :param str grouper_name: Name for the column to be used as grouper.
    :param int threshold: Size of volume aggregated in bars to be compiled.
    :return:
    """

    current_vol = 0
    bar_counter = 0
    rows_with_bar_counter = []

    for idx, row in trades.iterrows():
        new_data = dict(row.to_dict())
        new_data.update({grouper_name: bar_counter})
        rows_with_bar_counter.append(new_data)
        current_vol += row[agg_column]

        # render new row
        if current_vol >= threshold:
            bar_counter += 1
            current_vol = 0

    return pd.DataFrame(data=rows_with_bar_counter, index=trades.index)


############
# df utils #
############


def time_index_from_timestamps(data: pd.DataFrame, timezone: str = 'Europe/Madrid', drop_col: bool = False):
    """
    Assumes existing timestamp column or at least Open timestamp column.

    :param pd.Dataframe data: A dataframe.
    :param bool drop_col: Drop column applied as index.
    :param str timezone: The index of the pandas dataframe in the object can be converted to any timezone, i.e. "Europe/Madrid"
                           - TZ database: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    :return pd.DataFrame: A dataframe copy with the new index. Timestamp columns will be not dropped.
    """
    df_ = data.copy(deep=True)
    time_cols = sorted([c for c in df_.columns if 'timestamp' in c.lower()])
    time_col = time_cols[0]
    df_.sort_values(time_col, inplace=True)
    df_.index = pd.DatetimeIndex(pd.to_datetime(df_[time_col], unit='ms')).tz_localize('UTC').tz_convert(timezone)

    if drop_col:
        df_.drop(time_col, axis=1, inplace=True)
    return df_


def columns_restriction(data: pd.DataFrame, mode: str, extra=None) -> pd.DataFrame:
    """
    Filter columns by preset.

    :param pd.DataFrame data: A dataframe.
    :param str mode: Presets are: TB, VB, DB, VIB, DIB, TRB, VRB, DRB. All from chapter 2 of AFML book (Marcos López de Prado).
        https://www.amazon.com/-/es/Marcos-Lopez-Prado/dp/1119482089
    :param list extra: Optional extra columns.
    :return pd.DataFrame: Just preset columns.
    """
    valid_modes = ['TB', 'VB', 'DB', 'IB', 'VIB', 'DIB', 'TRB', 'VRB', 'DRB']
    try:
        assert mode in valid_modes
    except AssertionError:
        raise BinPanException(f"BiPan Exception: {mode} not a valid type ---> Valid types = {valid_modes}")
    df = data.copy(deep=True)
    if extra is None:
        extra = []
    bool_cols = [c for c in data.columns if c.endswith('True') or c.endswith('False')]  # add created from bool columns
    cols = []
    if mode == 'TB' or mode == 'VB' or mode == 'DB' or mode == 'IB':
        cols = ['Open', 'High', 'Low', 'Close', 'Timestamp'] + bool_cols + extra
    return df[cols]


def generate_volume_column(data: pd.DataFrame, add_cols: tuple, quote_column: bool = None) -> pd.DataFrame:
    """
    Add two columns to generate volume of base.

    :param pd.DataFrame data: A dataframe.
    :param tuple add_cols: Ordered tuple with two columns to add for calculating the total volume.
    :param bool quote_column: If True, Volume for quote will be added obtained from Volume and Close price.
    :return pd.DataFrame: A copy with volume added.
    """
    df = data.copy(deep=True)
    df[add_cols[0]].fillna(value=0, inplace=True)
    df[add_cols[1]].fillna(value=0, inplace=True)
    df.loc[:, 'Volume'] = df.loc[:, add_cols[0]] + df.loc[:, add_cols[1]]
    if quote_column:
        df.loc[:, 'Volume quote'] = df['Close'] * df['Volume']
    return df


############
# concepts #
############

def sign_of_price(data: pd.DataFrame, col_name: str = 'sign') -> pd.DataFrame:
    """
    Creates a new dataframe with a column with the sign of the price by each trade.
    1 if the trade increased the price and -1 for decreased price. Also called "tick rule".

    Nans will be filled as zero.

    .. math::

       b_t= \\begin{cases}b_{t-1} & \\text { if } \\Delta p_t=0 \\\\ \\frac{\\left|\\Delta p_t\\right|}{\\Delta p_t} & \\text { if } \\Delta p_t \\neq 0\\end{cases}


    :param pd.DataFrame data: Trades dataframe.
    :param str col_name: New column name.
    :return pd.DataFrame: A copy of the data with the new column.

    """
    df = data.copy(deep=True)
    df.loc[:, col_name] = np.sign(df['Price'].diff()).fillna(0)
    return df


##################
# AFML shortcuts #
##################


def tick_bars(trades: pd.DataFrame, size: int) -> pd.DataFrame:
    """
    Creates Tick Bars OHLC bars from trades-

    :param pd.DataFrame trades: Expected binpan aggregated trades or atomic trades dataframe.
    :param int size: Size of ticks in bars to be compiled.
    :return: A dataframe sampled with the new bars sampling.
    """
    if trades.empty:
        raise BinPanException("BinPan Exception: Tick Bars cannot be calculated with empty data.")
    df = generate_count_grouper_column(data=trades, grouper_name='group', size=size)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(df, 'Quantity', 'Buyer was maker', 'group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='TB')
    df = time_index_from_timestamps(df)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]))
    return df


def volume_bars(trades: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Creates Volume Bars OHLC bars from trades.

    :param pd.DataFrame trades: Expected binpan aggregated trades or atomic trades dataframe.
    :param int threshold: Size of volume threshold in bars to be compiled.
    :return: A dataframe sampled with the new bars sampling.
    """
    if trades.empty:
        raise BinPanException("BinPan Exception: Volume Bars cannot be calculated with empty data.")
    df = tag_by_accumulation(trades=trades, threshold=threshold, agg_column='Quantity', grouper_name='group')
    # df = generate_count_grouper_column(data=trades, grouper_name='group', size=size)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='VB')
    df = time_index_from_timestamps(df)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]))
    return df


def dollar_bars(trades: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Creates Dollar (or quote) Bars OHLC bars from trades.

    :param pd.DataFrame trades: Expected binpan aggregated trades or atomic trades dataframe.
    :param int threshold: Size of Dollar (or quote) threshold in bars to be compiled.
    :return: A dataframe sampled with the new bars sampling.
    """
    if trades.empty:
        raise BinPanException("BinPan Exception: Dollar Bars cannot be calculated with empty data.")
    df = trades.copy(deep=True)
    try:
        assert 'Quote quantity' in trades.columns
    except AssertionError:
        print("Added quote by product.")
        df.loc[:, 'Quote quantity'] = df['Price'] * df['Quantity']
    df = tag_by_accumulation(trades=df, threshold=threshold, agg_column='Quote quantity', grouper_name='group')
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='DB')
    df = time_index_from_timestamps(df)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]), quote_column=True)
    return df


def ohlc_bars(rows_with_bar_counter: list, index: pd.Index, mode: str = 'IB'):
    """
    Generate an OHLC dataframe from a trades dataframe with a column containing sampled numeric tags for each bar's aggregation.

    :param list rows_with_bar_counter: A list of trades with a column containing bar numbers to group into OHLC bars.
    :param pd.Index index: Index for the resulting dataframe.
    :param str mode: Presets are: TB, VB, DB, VIB, DIB, TRB, VRB, DRB. All from chapter 2 of AFML book (Marcos López de Prado).
        https://www.amazon.com/-/es/Marcos-Lopez-Prado/dp/1119482089
    :return: A dataframe with OHLC bars.
    """

    df = pd.DataFrame(data=rows_with_bar_counter, index=index)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode=mode)
    df = time_index_from_timestamps(df)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]), quote_column=True)
    return df


class ImbalanceBars(object):
    """
    Sample imbalance bars.

    :param pd.DataFrame trades: A dataframe with BinPan trades, atomic or aggregated.
    :param str bar_type: Can be 'imbalance', 'volume', 'dollar'.
    :param str method: Expected imbalance calculation method can be 'fix', 'sma', 'ema'
    :param int threshold: A threshold for fixed imbalance
    :param int window: A rolling window size for moving averages.
    """

    def __init__(self,
                 trades: pd.DataFrame,
                 bar_type: str,
                 method: str = 'ema',
                 threshold: int = 10,
                 window: int = 21,
                 boot_trades: int = 50):
        assert method in ['fix', 'sma', 'ema']
        assert bar_type in ['imbalance', 'volume', 'dollar']

        # self.trades = handlers.market.convert_to_numeric(sign_of_price(data=trades, col_name='sign'), mode='ignore')
        self.trades = sign_of_price(data=trades, col_name='sign')

        self.bar_type = bar_type
        self.method = method
        self.threshold = threshold
        self.window = window

        self.rows_with_bar_counter = []
        self.sampled_sizes = pd.Series(dtype=float)
        self.sampled_probabilities = pd.Series(dtype=float)  # 2.3.2.1

        # init startup variables

        if method == 'fix':
            self.boot_trades = 0
            self.molecule = pd.Series(dtype=float)
            self.current_imbalance = threshold
            self.current_probability = None
            self.expected_imbalance = self.threshold
        else:
            self.boot_trades = boot_trades
            self.molecule = self.get_molecule(length=self.boot_trades, pointer=self.boot_trades)
            self.current_imbalance = self.get_imbalance(my_molecule=self.molecule)
            self.current_probability = self.get_probability(my_molecule=self.molecule)
            self.expected_imbalance = self.current_imbalance

        self.bar_counter = self.boot_trades

        # print(self.molecule.reset_index())
        print("current_imbalance", self.current_imbalance)
        print("current_probability", self.current_probability)

        # main loop
        self.sampling_loop()
        index = self.trades.iloc[self.boot_trades:].index
        self.bars = ohlc_bars(self.rows_with_bar_counter, index=index, mode='IB')  # todo: check another modes for DIB or VIB

    def get_molecule(self, length: int, pointer: int):  # for starting the loop with initiated vars
        return self.trades.iloc[pointer - length:pointer]

    def get_expected_size(self) -> float:
        if self.method == "ema":
            return self.sampled_sizes.ewm(span=self.window).mean().iloc[-1]
        elif self.method == "sma":
            # return self.sampled_sizes.sma(window=self.window).mean().iloc[-1]
            return self.sampled_sizes.rolling(window=self.window).mean().iloc[-1]

    def get_expected_probability(self) -> float:  # 2.3.2.1
        if self.method == "ema":
            ret = self.sampled_probabilities.ewm(span=self.window).mean().iloc[-1]
        elif self.method == "sma":
            ret = self.sampled_probabilities.sma(window=self.window).mean().iloc[-1]
        else:
            print("method:", self.method)
            raise Exception("el método se pira")
        if ret:
            return ret
        return 0

    def get_expected_imbalance(self) -> float or int:
        if self.bar_type == 'imbalance':
            if self.method == "fix":
                return self.threshold
            else:
                return self.get_expected_size() * self.get_expected_probability()
        else:
            # TODO: for other imbalance bar types
            pass

    def get_imbalance(self, my_molecule: pd.DataFrame):
        if self.bar_type == "imbalance":
            # ret = np.sign(my_molecule['Price'].diff()).fillna(0).sum()
            ret = my_molecule['sign'].sum()
        else:  # fixme: nuevos tipos de bars
            ret = None
        if ret:
            return ret
        return 0

    def get_probability(self, my_molecule: pd.DataFrame):
        positive_signs = my_molecule['sign'].value_counts().get(1, 0)
        if self.bar_type == 'imbalance':
            ret = (2 * positive_signs / len(my_molecule)) - 1
        else:  # fixme: nuevos tipos de bars
            ret = None
        if ret:
            return ret
        return 0

    def sampling_loop(self):
        my_molecule = pd.DataFrame(columns=self.trades.columns)

        for idx, row in tqdm(self.trades.iloc[self.boot_trades:].iterrows()):

            # recent data
            new_data = dict(row.to_dict())
            new_data.update({'group': self.bar_counter})
            self.rows_with_bar_counter.append(new_data)

            # new_data.update({'group': bar_counter})
            # my_molecule = my_molecule.append(new_data, ignore_index=True)
            bool_columns = [col for col in pd.DataFrame([new_data]).columns if
                            pd.DataFrame([new_data])[col].dtype == 'object' and pd.DataFrame([new_data])[col].apply(
                                lambda x: isinstance(x, bool)).all()]
            new_dataframe = pd.DataFrame([new_data]).astype({col: bool for col in bool_columns})

            my_molecule = pd.concat([my_molecule, new_dataframe], ignore_index=True)
            # my_molecule = pd.concat([my_molecule, pd.DataFrame([new_data]).astype(bool)], ignore_index=True)

            # Explicitly cast the columns with all-bool values to bool dtype
            for col in bool_columns:
                new_dataframe[col] = new_dataframe[col].astype(bool)

            # metrics for expected values
            self.current_imbalance = self.get_imbalance(my_molecule=my_molecule)
            # self.current_probability = self.get_probability(my_molecule=my_molecule)

            # sample or not
            my_molecule = self.decide_sampling(my_molecule)

    def decide_sampling(self, my_molecule: pd.DataFrame):
        if abs(self.current_imbalance) >= self.expected_imbalance:
            self.bar_counter += 1
            # self.sampled_sizes = self.sampled_sizes.append(len(my_molecule), ignore_index=True)
            self.sampled_sizes = pd.concat([self.sampled_sizes, pd.Series([len(my_molecule)])], ignore_index=True)

            self.current_probability = self.get_probability(my_molecule=my_molecule)

            # self.sampled_probabilities = self.sampled_probabilities.append(self.current_probability, ignore_index=True)
            self.sampled_probabilities = pd.concat([self.sampled_probabilities, pd.Series([self.current_probability])], ignore_index=True)
            self.expected_imbalance = self.get_expected_imbalance()

            my_molecule = pd.DataFrame(columns=self.trades.columns)
            self.current_imbalance = 0
        return my_molecule


def imbalance_bars_divergent(trades: pd.DataFrame, starting_imbalance: float) -> pd.DataFrame:
    """
    Generates candles by grouping each accumulated imbalance defined by:

    .. math::

       \\theta_T=\\sum_{t=1}^T b_t

    To close a bar, the imbalance must meet expected imbalance while iterating trades, in other words, when expected ticks times the
    difference between probability of positive signs versus negative signes meets the imbalance.

    .. math::

       T^*=\\underset{T}{\\arg \\min }\\left\\{\\left|\\theta_T\\right| \\geq \\mathrm{E}_0[T]\\left|2 \\mathrm{P}\\left[b_t=1\\right]-1\\right|\\right\\}

    .. note::

       No matter how you obtain expected ticks size and probability for the next imbalance threshold. It explodes. I will focus on fixed
       threshold in other function.

    :param pd.DataFrame trades: Expected binpan aggregated trades or atomic trades dataframe.
    :param float starting_imbalance: Starting value for following bars. Its recommended to wait some bars quantity to consider established sizes.
    :return: A dataframe sampled with the new bars sampling.
    """
    df = sign_of_price(data=trades, col_name='sign')
    current_imbalance = 0
    bar_counter = 0
    rows_with_bar_counter = []

    current_trades_qty = 0
    positive_qty = 0
    expected_imbalance = starting_imbalance

    for idx, row in df.iterrows():

        # save data with bar counter
        new_data = dict(row.to_dict())
        new_data.update({'group': bar_counter})
        rows_with_bar_counter.append(new_data)

        # metrics for closing the current bar
        sign = new_data['sign']
        current_imbalance += sign

        # expected values for next trade
        current_trades_qty += 1
        if sign > 0:
            positive_qty += 1
        prob = (2 * positive_qty / current_trades_qty) - 1

        # update values if closed bar
        if abs(current_imbalance) >= expected_imbalance:
            bar_counter += 1
            # just previous values for now, weighted expected values will diverge too.
            print(
                f"Current imbalance:{current_imbalance} trades:{current_trades_qty} pos_trades:{positive_qty} prob:{prob} expected_imbalance"
                f":{abs(current_trades_qty * prob)}")
            expected_imbalance = abs(current_trades_qty * prob)
            current_trades_qty = 0
            current_imbalance = 0
            positive_qty = 0

    df = pd.DataFrame(data=rows_with_bar_counter, index=trades.index)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='IB')
    df = time_index_from_timestamps(df)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]), quote_column=True)
    return df


def imbalance_bars_fixed(trades: pd.DataFrame, imbalance: float) -> pd.DataFrame:
    """
    Generates candles by grouping each accumulated fixed imbalance threshold defined by:

    .. math::

       \\theta_T=\\sum_{t=1}^T b_t

    To close a bar, the imbalance must meet expected fixed imbalance while iterating trades.

    .. math::

       T^*=\\underset{T}{\\arg \\min }\\left\\{\\left|\\theta_T\\right| \\geq \\mathrm{E}_0[T]\\left|2 \\mathrm{P}\\left[b_t=1\\right]-1\\right|\\right\\}

    .. note::

       Fixed threshold for rendering imbalance bars.

    :param pd.DataFrame trades: Expected binpan aggregated trades or atomic trades dataframe.
    :param float imbalance: Fixed value for imbalance threshold.
    :return: A dataframe sampled with the new bars sampling.
    """
    df_ = sign_of_price(data=trades, col_name='sign')
    current_imbalance = 0
    bar_counter = 0
    rows_with_bar_counter = []

    current_trades_qty = 0
    positive_qty = 0

    for idx, row in df_.iterrows():

        # save data with bar counter
        new_data = dict(row.to_dict())
        new_data.update({'group': bar_counter})
        rows_with_bar_counter.append(new_data)

        # metrics for closing the current bar
        sign = new_data['sign']
        current_imbalance += sign

        # expected values for next trade
        current_trades_qty += 1
        if sign > 0:
            positive_qty += 1
        prob = (2 * positive_qty / current_trades_qty) - 1

        # update values if closed bar
        if abs(current_imbalance) >= imbalance:
            bar_counter += 1
            # just previous values for now
            print(
                f"Current imbalance:{current_imbalance} trades:{current_trades_qty} pos_trades:{positive_qty} prob:{prob} expected_imbalance"
                f":{abs(current_trades_qty * prob)}")
            current_trades_qty = 0
            current_imbalance = 0
            positive_qty = 0

    df = pd.DataFrame(data=rows_with_bar_counter, index=trades.index)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='IB')
    df = time_index_from_timestamps(df)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]), quote_column=True)
    return df
