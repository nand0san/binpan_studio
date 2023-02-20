"""
Data Aggregation.
"""
import pandas as pd


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
    Creates  OHLC columns for a column based on group by other column values.

    :param pd.Dataframe data: A dataframe with at least two columns.
    :param str column_to_ohlc: Column to sparse values to OHLC columns in each group.
    :param str group_column: This column will be the grouping key.
    :return: A copy of the dataframe with OHLC data.
    """
    df = data.copy(deep=True)
    df['High'] = df.groupby([group_column])[column_to_ohlc].transform('max')
    df['Low'] = df.groupby([group_column])[column_to_ohlc].transform('min')
    df['Open'] = df.groupby([group_column])[column_to_ohlc].transform('first')
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
    df.fillna(inplace=True, method='ffill')
    # df.fillna(inplace=True, method='bfill')
    # fixme: strage assignement for the first or last each streak
    return df[cols]


def aggregate_by(data: pd.DataFrame, group_column: str, by='last') -> pd.DataFrame:
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


def aggregate_last(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Drop lines except the first row of each group_column streak.

    :param pd.Dataframe data: A dataframe.
    :param str group_column: This column will be the grouping key.
    :return: A copy of the dataframe with just the first row each grouper streak.
    """
    df = data.copy(deep=True)
    aggregator = {c: 'last' for c in df.columns}
    return df.groupby(group_column).agg(aggregator)