"""
Data Aggregation.
"""
import pandas as pd


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
