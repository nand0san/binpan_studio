import pandas as pd
from random import sample
from .logs import Logs


tags_logger = Logs(filename='./logs/tags_logger.log', name='tags_logger', info_level='INFO')


def random_strategy(data: pd.DataFrame, buys_qty: int = 10, sells_qty: int = 10, new_actions_col: str = 'actions'):
    """
    Creates a random buy and sell tag in a column of a dataframe.

    :param pd.DataFrame data: A dataframe.
    :param int buys_qty: Buy tags quantity. Default is 10.
    :param int sells_qty: Sells tags quantity. Default is 10.
    :param str new_actions_col: Name of the new column with the buy and sell tags. Defaults to 'actions'
    :return: pd.DataFrame

    """
    df = data.copy(deep=True)
    buys = sample(list(df.index), buys_qty)
    sells = sample(list(df.index), sells_qty)
    df.loc[buys, new_actions_col] = 'buy'
    df.loc[sells, new_actions_col] = 'sell'
    df[new_actions_col].fillna('-', inplace=True)
    return df
