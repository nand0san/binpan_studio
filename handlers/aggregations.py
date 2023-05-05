"""
Data Aggregation.
"""
import pandas as pd
import numpy as np
from typing import Tuple

from .exceptions import BinPanException
from .starters import is_python_version_numba_supported

if is_python_version_numba_supported():
    from .stat_tests import ema_numba, sma_numba
else:
    from .stat_tests import ema_numpy as ema_numba, sma_numpy as sma_numba


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


def sum_split_by_boolean_column_and_group(data: pd.DataFrame, column_to_split_sum: str = "Quantity", bool_col: str = "Buyer was maker",
                                          group_column: str = "group") -> pd.DataFrame:
    """
    Splits the sum of a numeric column into two new columns based on a boolean column and groups the data using another column.

    This function is useful for generating two new columns, one for the true values and another for the false values, with the sum
    of the specified numeric column for each group and filling the remaining cells with NaNs.

    :param data: A dataframe with at least 3 columns.
    :type data: pd.DataFrame
    :param column_to_split_sum: Numeric column with data to sum by group and split by boolean column, defaults to "Quantity".
    :type column_to_split_sum: str, optional
    :param bool_col: Column to define splitting, defaults to "Buyer was maker".
    :type bool_col: str, optional
    :param group_column: This column will be the grouping key, defaults to "group".
    :type group_column: str, optional
    :return: A copy of the dataframe with splitted sum by group from the column to split sum.
    :rtype: pd.DataFrame
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


def count_trues_cumulative(data: pd.DataFrame, bool_column: str, new_column: str) -> pd.DataFrame:
    """
    Add a new column to the DataFrame with cumulative numbers for each True value in a boolean column.

    :param data: The input DataFrame.
    :type data: pd.DataFrame
    :param bool_column: The name of the boolean column to evaluate.
    :type bool_column: str
    :param new_column: The name of the new column to store the cumulative numbers for True values.
    :type new_column: str
    :return: A new DataFrame with an additional column containing the cumulative numbers for True values.
    :rtype: pd.DataFrame
    """
    df = data.copy(deep=True)
    counter = 0

    def increment_if_true(x):
        nonlocal counter
        if x:
            counter += 1
        return counter

    df[new_column] = df[bool_column].apply(increment_if_true)
    return df


def drop_aggregated(data: pd.DataFrame, group_column: str, by='last') -> pd.DataFrame:
    """
    Drop lines except the first/last/min/max etc row of each group_column streak.

    It assumes that the grouping column is a serie of integers to group it by blocks.

    :param pd.Dataframe data: A dataframe.
    :param str group_column: This column will be the grouping key.
    :param str by: Pandas valid aggregation method.
    :return: A copy of the dataframe with just the first row each grouper streak.
    """
    df = data.copy(deep=True)

    # type test
    is_int_or_nan = df[group_column].apply(lambda x: isinstance(x, (int, np.integer)) or np.isnan(x))
    assert is_int_or_nan.all(), "Group column must use integer numbers"

    # sequential integers fault warning
    integers = data.loc[df[group_column].apply(lambda x: isinstance(x, (int, np.integer))), group_column]
    differences = np.diff(integers)
    differences = np.isnan(differences) | (differences == 1)
    try:
        assert differences.all(), f"Warning: Numbers in column '{group_column}' are not consecutive."
    except:
        pass

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


def time_index_from_timestamps(data: pd.DataFrame, index_name: str = None, timezone: str = 'Europe/Madrid', drop_col: bool = False, ):
    """
    Assumes existing timestamp column or at least Open timestamp column.

    :param pd.Dataframe data: A dataframe.
    :param bool drop_col: Drop column applied as index.
    :param str index_name: Name for the resulting index.
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
    if index_name:
        df_.index.name = index_name
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
    # df[col_name] = df['Price'].diff()

    df[col_name] = df['Price'].diff().astype(float)

    df[col_name] = np.where(df[col_name] == 0., np.nan, np.sign(df[col_name]))
    df[col_name].fillna(method='ffill', inplace=True)
    df[col_name] = df[col_name].fillna(0).astype(int)
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
    index_name = trades.index.name

    df = generate_count_grouper_column(data=trades, grouper_name='group', size=size)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(df, 'Quantity', 'Buyer was maker', 'group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='TB')
    df = time_index_from_timestamps(df, index_name=index_name)
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
    index_name = trades.index.name

    df = tag_by_accumulation(trades=trades, threshold=threshold, agg_column='Quantity', grouper_name='group')
    # df = generate_count_grouper_column(data=trades, grouper_name='group', size=size)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='VB')
    df = time_index_from_timestamps(df, index_name=index_name)
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
    index_name = trades.index.name

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
    df = time_index_from_timestamps(df, index_name=index_name)
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
    index_name = df.index.name

    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode=mode)
    df = time_index_from_timestamps(df, index_name=index_name)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]), quote_column=True)
    return df


class ImbalanceBars(object):
    """
    Sample imbalance bars.

    :param pd.DataFrame trades: A dataframe with trades, atomic or aggregated.
    :param str bar_type: Can be 'imbalance', 'volume', 'dollar'.
    :param str method: Expected imbalance calculation method can be 'fix', 'sma', 'ema'
    :param int fixed_imbalance_threshold: A threshold for fixed imbalance
    :param int window: A rolling window size for moving averages.
    :param int boot_trades: Number of trades to use for initial imbalance calculation.
    :param bool verbose: Prints data while computing bars.
    :param bool adjust_threshold: If true, threshold will be multiplied by first price. Ca be useful with volume bars.
    """

    def __init__(self, trades: pd.DataFrame, bar_type: str, method: str = 'ema', fixed_imbalance_threshold: int = 1000, window: int = 21,
                 boot_trades: int = 1000, verbose: bool = False, adjust_threshold: bool = False):
        """
        Initialize the ImbalanceBars object with the given parameters.
        """
        assert method in ['fix', 'sma', 'ema']
        assert bar_type in ['imbalance', 'volume', 'dollar']

        self.trades = sign_of_price(data=trades, col_name='sign')
        self.bar_type = bar_type
        self.method = method
        self.threshold = fixed_imbalance_threshold
        self.window = window
        self.boot_trades = boot_trades
        self.verbose = verbose
        self.adjust_threshold = adjust_threshold

        self.rows_with_bar_counter = []
        self.sampled_sizes = np.empty(shape=(0,), dtype=float)
        self.sampled_probabilities = np.empty(shape=(0,), dtype=float)

        self.current_size = None
        self.current_imbalance = None
        self.expected_imbalance = None
        self.bar_counter = 0
        self.bars = None

        self.initialize_startup_variables()
        self.sampling_loop()
        self.construct_bars()

    def initialize_startup_variables(self):
        """
        Initialize startup variables based on the chosen method.
        """

        if self.method == 'fix':
            if self.adjust_threshold:
                print(f"Threshold adjusted by price: {self.threshold} ---> {self.threshold * self.trades.iloc[0]['Price']}")
                self.expected_imbalance = self.threshold * self.trades.iloc[0]['Price']
            else:
                self.expected_imbalance = self.threshold
            current_probability = 0
        else:
            self.current_size = self.boot_trades

            if self.bar_type == 'imbalance':
                current_probability = self.trades.iloc[:self.boot_trades]['sign'].sum()

            elif self.bar_type == "volume":
                current_probability = (self.trades.iloc[:self.boot_trades]['sign'] * self.trades.iloc[:self.boot_trades]['Quantity']).sum()

            elif self.bar_type == "dollar":
                current_probability = (self.trades.iloc[:self.boot_trades]['sign'] * self.trades.iloc[:self.boot_trades]['Quantity'] *
                                       self.trades.iloc[:self.boot_trades]['Price']).sum()
            else:
                raise BinPanException(f"Bar type implementation error: {self.bar_type}")

            self.expected_imbalance = self.current_size * abs(current_probability)

        if self.verbose:
            print(f"Boot current_probability: {current_probability}")
            print(f"Boot current_size: {self.boot_trades}")
            print(f"Boot expected_imbalance: {self.expected_imbalance}")

    def get_expected_size(self) -> float:
        """
        Calculate the expected size of the imbalance bars using the chosen method.
        """
        if self.method == "ema":
            return ema_numba(self.sampled_sizes, window=self.window)[-1]
        elif self.method == "sma":
            return sma_numba(self.sampled_sizes, window=self.window)[-1]

    def get_expected_probability(self) -> float:
        """
        Calculate the expected probability of the imbalance bars using the chosen method.
        """
        ret = None
        if self.method == "ema":
            ret = ema_numba(self.sampled_probabilities, window=self.window)[-1]
        elif self.method == "sma":
            ret = sma_numba(self.sampled_probabilities, window=self.window)[-1]
        return ret

    def get_expected_imbalance(self) -> float or int:
        """
        Calculate the expected imbalance value based on the bar type and chosen method.
        """
        if self.method == "fix":
            return self.threshold
        else:
            size = self.get_expected_size()
            exp_prob = self.get_expected_probability()
            return abs(size * exp_prob)

    def sampling_loop(self) -> None:
        """
        Main sampling loop to iterate through trades and create imbalance bars.
        """
        my_molecule_sign = []
        my_molecule_cum_size = 0

        for idx, row in self.trades.iterrows():
            new_data = dict(row.to_dict())

            my_molecule_cum_size += 1

            if self.bar_type == 'imbalance':
                my_molecule_sign += [new_data['sign']]
            elif self.bar_type == 'volume':
                my_molecule_sign += [new_data['sign'] * new_data['Quantity']]
            else:  # dollar imbalance bars
                my_molecule_sign += [new_data['sign'] * new_data['Quantity'] * new_data['Price']]

            my_molecule_sign, my_molecule_cum_size = self.decide_sampling(my_molecule_sign, my_molecule_cum_size)

            new_data.update({'group': self.bar_counter})
            self.rows_with_bar_counter.append(new_data)

    def decide_sampling(self, my_molecule_sign: list, my_molecule_cum_size: int) -> Tuple[list, int]:
        """
        Decide whether to sample the current bar and update variables accordingly.

        Sample surpassing data to the next bar.
        """
        prob = abs(np.sum(np.array(my_molecule_sign)))
        self.current_imbalance = abs(my_molecule_cum_size * prob)

        # check
        # print(self.current_imbalance, self.expected_imbalance)
        if self.current_imbalance >= self.expected_imbalance:

            if self.verbose:
                print(f"exp_imb:{self.expected_imbalance:.8f}\tsize:{my_molecule_cum_size}\tprob:{prob:.8f}\timbalance"
                      f":{self.current_imbalance}")

            self.bar_counter += 1
            self.sampled_sizes = np.append(self.sampled_sizes, my_molecule_cum_size - 1)  # -1 just prev size, ojo q da negativo si viene

            # modo practico sin ultimo valor opcional
            prob = abs(np.sum(np.array(my_molecule_sign[:-1])))  # just prev values [:-1]

            self.sampled_probabilities = np.append(self.sampled_probabilities, prob)  # just prev values

            my_molecule_cum_size = 0
            my_molecule_sign = []

            self.expected_imbalance = self.get_expected_imbalance()

        return my_molecule_sign, my_molecule_cum_size

    def construct_bars(self) -> None:
        """
        Construct the resulting imbalance bars DataFrame.
        """
        my_index = self.trades.index
        self.bars = ohlc_bars(self.rows_with_bar_counter, index=my_index, mode='IB')


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
    index_name = trades.index.name
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
            print(f"Current imbalance:{current_imbalance} trades:{current_trades_qty} pos_trades:{positive_qty} prob:{prob} expected_imbalance"
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
    df = time_index_from_timestamps(df, index_name=index_name)
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
    index_name = trades.index.name
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
            print(f"Current imbalance:{current_imbalance} trades:{current_trades_qty} pos_trades:{positive_qty} prob:{prob} expected_imbalance"
                  f":{abs(current_trades_qty * prob)}")
            current_trades_qty = 0
            current_imbalance = 0
            positive_qty = 0

    df = pd.DataFrame(data=rows_with_bar_counter, index=trades.index)
    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='group')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='group')
    df = drop_aggregated(data=df, group_column='group', by='first')
    df = columns_restriction(data=df, mode='IB')
    df = time_index_from_timestamps(df, index_name=index_name)
    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]), quote_column=True)
    return df


def tick_imbalance_bars_chat_gpt(trades: pd.DataFrame, window: int = 10):
    df = trades.copy(deep=True)
    index_name = df.index.name

    # tick rule
    df['delta_p'] = df['Price'].diff()
    df['b_t'] = np.where(df['delta_p'] == 0, np.nan, np.sign(df['delta_p']))
    df['b_t'].fillna(method='ffill', inplace=True)

    df['theta_T'] = df['b_t'].cumsum()

    # milliseconds T
    df['T'] = (df.index.to_series().diff().dt.total_seconds() * 1000).fillna(0).astype(int)

    # expectation
    df['E0_T'] = df['T'].ewm(span=window).mean()
    df['E0_P'] = df['b_t'].ewm(span=window).mean()
    df['expected_imbalance'] = np.abs(df['E0_T'] * (2 * df['E0_P'] - 1))

    # sampling
    df['T_star'] = np.abs(df['theta_T']) >= np.abs(df['E0_T'] * (2 * df['E0_P'] - 1))

    df = ohlc_group(data=df, column_to_ohlc='Price', group_column='T_star')
    df = sum_split_by_boolean_column_and_group(data=df, column_to_split_sum='Quantity', bool_col='Buyer was maker', group_column='T_star')
    df = count_trues_cumulative(data=df, bool_column='T_star', new_column="T_start_count")

    df = drop_aggregated(data=df, group_column='T_start_count', by='first')

    df = columns_restriction(data=df, mode='IB', extra=['expected_imbalance', 'E0_T', 'E0_P'])
    df = time_index_from_timestamps(df, index_name=index_name)

    # add Volume for plotting
    bool_cols = [c for c in df.columns if 'True' in c or 'False' in c]
    df = generate_volume_column(data=df, add_cols=(bool_cols[0], bool_cols[1]), quote_column=True)
    return df
