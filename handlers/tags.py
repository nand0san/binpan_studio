import pandas as pd
import numpy as np


def tag_value(serie: pd.Series,
              value: int or float or dd,
              gt: bool = False,
              ge: bool = False,
              eq: bool = False,
              le: bool = False,
              lt: bool = False,
              match_tag=1,
              mismatch_tag=0,
              to_numeric=True
              ) -> pd.Series:
    """
    It tags values of a serie compared to one value and a gt,ge,eq,le,lt condition.

    :param pd.Series serie: A numeric serie.
    :param int or float or Decimal value: Value of comparison.
    :param bool gt: Value to match if strictly greater than value.
    :param bool ge: Value to match if greater or equal than value.
    :param bool eq: Value to match if equal than value.
    :param bool le: Value to match if lower or equal than value.
    :param bool lt: Value to match if strictly lower than value.
    :param int or str match_tag: Value to tag previous logic.
    :param mismatch_tag: Value to tag not matched previous logic.
    :param bool to_numeric: If true, if possible, downcast type until the most basic numeric type (integer)
    :return pd.Series: A serie with tags as values.
    """
    ret = pd.Series(index=serie.index, dtype=type(match_tag))

    if gt:
        ret = serie.gt(value)
    if ge:
        ret = serie.ge(value)
    if eq:
        ret = serie.eq(value)
    if le:
        ret = serie.le(value)
    if lt:
        ret = serie.lt(value)

    ret.loc[ret] = match_tag
    ret.loc[ret == False] = mismatch_tag
    if to_numeric:
        return pd.to_numeric(arg=ret, downcast='integer')
    else:
        return ret


def tag_comparison(serie_a: pd.Series,
                   serie_b: pd.Series,
                   gt: bool = False,
                   ge: bool = False,
                   eq: bool = False,
                   le: bool = False,
                   lt: bool = False,
                   match_tag=1,
                   mismatch_tag=0,
                   to_numeric=True
                   ) -> pd.Series:
    """
    It tags values of a serie compared to other serie by methods gt,ge,eq,le,lt condition.

    :param pd.Series serie_a: A numeric serie.
    :param pd.Series serie_b: A numeric serie.
    :param bool gt: Value to match if strictly greater than serie_b values.
    :param bool ge: Value to match if greater or equal than serie_b values.
    :param bool eq: Value to match if equal than serie_b values.
    :param bool le: Value to match if lower or equal than serie_b values.
    :param bool lt: Value to match if strictly lower than serie_b values.
    :param int or str match_tag: Value to tag previous logic.
    :param mismatch_tag: Value to tag not matched previous logic.
    :param bool to_numeric: If true, if possible, downcast type until the most basic numeric type (integer)
    :return pd.Series: A serie with tags as values.
    """
    ret = pd.Series(index=serie_a.index, dtype=type(match_tag))

    if gt:
        ret = serie_a.gt(serie_b)
    if ge:
        ret = serie_a.ge(serie_b)
    if eq:
        ret = serie_a.eq(serie_b)
    if le:
        ret = serie_a.le(serie_b)
    if lt:
        ret = serie_a.lt(serie_b)

    ret.loc[ret] = match_tag
    ret.loc[ret == False] = mismatch_tag

    if to_numeric:
        return pd.to_numeric(arg=ret, downcast='integer')
    else:
        return ret


def tag_cross(serie_a: pd.Series,
              serie_b: pd.Series,
              echo: int = 0,
              cross_over_tag='buy',
              cross_below_tag='sell',
              name='Cross',
              non_zeros: bool = True) -> pd.Series:
    """
    It tags points where serie_a crosses over serie_b or optionally below.

    :param pd.Series serie_a: A numeric serie.
    :param pd.Series serie_b: A numeric serie.
    :param int echo: It tags a fixed amount of candles forward the crossed point not including cross candle.
    :param int or str cross_over_tag: Value to tag over cross. Default is "buy".
    :param int or str cross_below_tag: Value to tag below cross. Default is "sell"
    :param str name: Name for the resulting serie.
    :param bool non_zeros: Substitutes zeros by nans. If echo want to be used, must be used non_zeros.
    :return pd.Series: A serie with tags as values.
    """
    dif = tag_comparison(serie_a=serie_a, serie_b=serie_b, gt=True)
    ret = dif.diff()

    if echo:
        ret = ret.ffill(limit=echo)

    ret.name = name
    if non_zeros:
        ret = ret
    else:
        ret = ret.replace({'1': cross_over_tag, 1: cross_over_tag, '-1': cross_below_tag, -1: cross_below_tag})

    if echo:
        ret = ret.ffill(limit=echo)
    return ret


################
# BACK TESTING #
################

def buy_base_backtesting(row: pd.Series, price: float, base: float, quote: float, fee=0.001) -> tuple:
    """
    A simple backtesting buy function.

    :param pd.Series row: A row of a BinPan DataFrame with prices.
    :param float price: Price of the action.
    :param float base: Inverted base.
    :param float quote: Inverted quote.
    :param float fee: Fee applied to the operation.
    :return tuple: A tuple with resulting base and quote after operation.
    """
    high_value = row['High']
    low_value = row['Low']

    if low_value <= price <= high_value:
        ordered_price = price

    elif price > high_value:
        ordered_price = high_value

    else:
        return base, quote

    base += (quote / ordered_price) * (1 - fee)
    quote = 0
    return base, quote


def sell_base_backtesting(row: pd.Series, price: float, base: float, quote: float, fee=0.001) -> tuple:
    """
    A simple backtesting sell function.

    :param pd.Series row: A row of a BinPan DataFrame with prices.
    :param float price: Price of the action.
    :param float base: Inverted base.
    :param float quote: Inverted quote.
    :param float fee: Fee applied to the operation.
    :return tuple: A tuple with resulting base and quote after operation.
    """
    high_value = row['High']
    low_value = row['Low']

    if low_value <= price <= high_value:
        ordered_price = price

    elif price < low_value:
        ordered_price = low_value

    else:
        return base, quote

    quote += base * ordered_price * (1 - fee)
    base = 0
    return base, quote


def backtesting(df: pd.DataFrame,
                actions: pd.Series or str = None,
                base: float = 0,
                quote: float = 1000,
                priced_actions_col: str = 'Open',
                fee: float = 0.001,
                label_buy='buy',
                label_sell='sell',
                suffix: str = '',
                action_candles_lag=1) -> pd.DataFrame:
    """
    Returns two pandas series as base wallet and quote wallet over time. Its expected a serie with strings like 'buy' or 'sell'.
    That serie is called the "actions". If it is available a column with the exact price of the actions, can be passed in
    actions_col parameter by column name or a pandas series, if not, Closed price column can be a good approximation.

    All actions will be considered to buy all base as possible or to sell all base as posible.

    :param pd.DataFrame df: A BinPan dataframe.
    :param pd.Series actions: A pandas series with buy or sell strings for simulate actions.
    :param float base: A starting quantity of symbol's base.
    :param float quote: A starting quantity of symbol's quote.
    :param pd.Series or str priced_actions_col: Column with the prices for the action emulation.
    :param float fee: Binance applicable fee for trading. DEfault is 0.001.
    :param str or int label_buy: A label consider as trade in trigger.
    :param str or int label_sell: A label consider as trade out trigger.
    :param int action_candles_lag: Candles needed to confirm an action from action tag. Usually one candle. Example,
       when an action like a cross of two EMA lines occur, it's needed to close that candle of the cross to confirm,
       then, nex candle can buy at open.
    :param str suffix: A suffix for the names of the columns.
    :return tuple: Two series with the base wallet and quote wallet funds in time.
    """

    df_ = df.copy(deep=True)

    if type(actions) == str:
        actions_ = df_[actions].copy(deep=True)
    else:
        actions_ = actions.copy(deep=True)
    actions_no_zeros = actions_[actions_ != 0]
    actions_labels = list(actions_no_zeros.dropna().value_counts().index)

    try:
        assert {label_buy, label_sell} == set(actions_labels)
    except AssertionError:
        try:
            assert {1, -1} == set(actions_labels)
            label_buy = 1
            label_sell = -1
        except AssertionError:
            raise Exception(f"BinPan Exception: Backtesting expected labels were not correctly specified: {actions_labels}")

    base_wallet, quote_wallet = [], []

    last_action = 0
    lag = action_candles_lag

    for index, row in df_.iterrows():
        curr_action = actions_[index]
        price = df_.loc[index, priced_actions_col]

        if curr_action == label_buy and last_action != 'buy':
            last_action = 'buy'
            # base, quote = buy_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            # last_action = label_buy

        elif curr_action == label_sell and last_action != 'sell':
            last_action = 'sell'
            # base, quote = sell_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            # last_action = label_sell

        if last_action == 'buy' and lag == 0:
            base, quote = buy_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            last_action = ''
            lag = action_candles_lag

        elif last_action == 'sell' and lag == 0:
            base, quote = sell_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            last_action = ''
            lag = action_candles_lag
        elif last_action == 'buy' or last_action == 'sell':
            lag -= 1

        base_wallet.append(base)
        quote_wallet.append(quote)

    base_serie = pd.Series(base_wallet, index=df_.index, name=f"Wallet_base{suffix}")
    quote_serie = pd.Series(quote_wallet, index=df_.index, name=f"Wallet_quote{suffix}")

    return pd.DataFrame([base_serie, quote_serie]).T
