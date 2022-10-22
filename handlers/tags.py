"""

Tagging utils.

"""
import pandas as pd
import numpy as np
from typing import Tuple

from .exchange import get_info_dic, get_bases_dic, get_quotes_dic
from .market import get_candles_by_time_stamps, parse_candles_to_dataframe


def tag_value(serie: pd.Series,
              value: int or float,
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
    :return pd.Series: A serie with tags (argument) as values.
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
        ret = ret.replace({'1': cross_over_tag,
                           1: cross_over_tag,
                           '-1': cross_below_tag,
                           -1: cross_below_tag,
                           '0': np.nan,
                           0: np.nan
                           })
    else:
        ret = ret.replace({'1': cross_over_tag, 1: cross_over_tag, '-1': cross_below_tag, -1: cross_below_tag})

    if echo:
        ret = ret.ffill(limit=echo)
    return ret


def tag_strategy_group(column: str,
                       group: str,
                       strategy_groups: dict) -> dict:
    """
    Tags a columns as part of a Strategy group of columns.

    :param str column: A column to tag with a strategy group.
    :param str group: Name of the group.
    :param str strategy_groups: The existing strategy groups.
    :return dict: Updated strategy groups of columns.

    """

    if not group in strategy_groups.keys():
        strategy_groups.update({group: [column]})
    else:
        group_columns = strategy_groups[group]
        group_columns.append(column)
        strategy_groups.update({group: group_columns})
    return strategy_groups


################
# merge series #
################


def merge_series(predominant: pd.Series,
                 other: pd.Series) -> pd.Series:
    """
    Predominant serie will be filled nans with values, if existing, from the other serie.
    
    Same kind of index needed.
    
    :param pd.Series predominant: A serie with nans to fill from other serie.
    :param pd.Series other: A serie to pick values for the nans.
    :return pd.Series: A merged serie. 
    """
    return predominant.combine_first(other=other)


def clean_in_out(serie: pd.Series,
                 df: pd.DataFrame,
                 in_tag=1,
                 out_tag=-1) -> pd.Series:
    """
    Balance from in tag to first out tag and discards any other tag until next in tag.

    :param pd.Series serie: A serie with in and out tags.
    :param pd.DataFrame df: BinPan dataframe.
    :param in_tag: Tag for in tags. Default is 1.
    :param out_tag: Tag for out tags. Default is -1.
    :return pd.Series: Clean serie with ech in with next out.
    """

    ret = pd.Series(index=serie.index)

    last_tag = np.nan

    for idx, value in serie.iteritems():

        if value == out_tag and last_tag != out_tag:
            ret[idx] = value
            last_tag = value

        elif value == in_tag and last_tag != in_tag:
            ret[idx] = value

        if value == in_tag or value == out_tag:
            last_tag = value

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


def evaluate_wallets(df_: pd.DataFrame,
                     base_serie: pd.Series,
                     quote_serie: pd.Series,
                     evaluating_quote: str,
                     info_dic: dict = None,
                     suffix: str = ''
                     ) -> pd.DataFrame:
    """
    Obtains total wallet values in a quote value through time.

    :param pd.DataFrame df_: A BinPan's dataframe.
    :param pd.Series base_serie: A wallet serie for symbol's base.
    :param pd.Series quote_serie: A wallet serie for symbol's quote.
    :param str evaluating_quote: a Binance valid quote to evaluate operations.
    :param dict info_dic: BinPan exchange info dict to extract information about quotes and bases of symbols.
    :param str suffix: A suffix for the names of the columns.
    :return pd.DataFrame: It returns a dataframe with base and quote series plus an evaluated serie in any quote coin of both wallets merged.
    """
    evaluating_quote = evaluating_quote.upper()

    original_index = df_.index
    symbol = original_index.name.split()[0]
    tick_interval = original_index.name.split()[1]
    df_.set_index('Open timestamp', inplace=True, drop=False)

    if not info_dic:
        info_dic = get_info_dic()
    bases = get_bases_dic(info_dic=info_dic)
    quotes = get_quotes_dic(info_dic=info_dic)

    base = bases[symbol]
    quote = quotes[symbol]

    if quote == evaluating_quote:
        evaluated_base_serie = df_['Close']
        evaluated_quote_serie = pd.Series(1, index=df_.index)
    else:
        evaluate_base_symbol = base + evaluating_quote
        evaluate_quote_symbol = quote + evaluating_quote

        start_timestamp = df_['Open timestamp'].iloc[0]
        end_timestamp = df_['Close timestamp'].iloc[-1]

        evaluated_base_list = get_candles_by_time_stamps(symbol=evaluate_base_symbol,
                                                         tick_interval=tick_interval,
                                                         start_time=start_timestamp,
                                                         end_time=end_timestamp)
        evaluated_base_df = parse_candles_to_dataframe(raw_response=evaluated_base_list,
                                                       symbol=evaluate_base_symbol,
                                                       tick_interval=tick_interval)
        evaluated_base_df.set_index('Open timestamp', inplace=True, drop=False)

        evaluated_quote_list = get_candles_by_time_stamps(symbol=evaluate_quote_symbol,
                                                          tick_interval=tick_interval,
                                                          start_time=start_timestamp,
                                                          end_time=end_timestamp)
        evaluated_quote_df = parse_candles_to_dataframe(raw_response=evaluated_quote_list,
                                                        symbol=evaluate_quote_symbol,
                                                        tick_interval=tick_interval)

        evaluated_quote_df.set_index('Open timestamp', inplace=True, drop=False)
        # try:
        #     assert df_.index == evaluated_base_df.index
        #     assert df_.index == evaluated_quote_df.index
        # except AssertionError:
        #     raise Exception(f"Index not matching original object index when evaluating quote price.")

        evaluated_base_serie = evaluated_base_df['Close']
        evaluated_quote_serie = evaluated_quote_df['Close']

    # apply qty for the price
    base_value = evaluated_base_serie * base_serie
    quote_value = evaluated_quote_serie * quote_serie

    # merge data
    merged = base_value + quote_value

    merged.name = f"Evaluated_{symbol}_{evaluating_quote}{suffix}"
    merged.index = original_index
    return pd.DataFrame([base_serie, quote_serie, merged]).T


def check_action_labels_for_backtesting(actions: pd.Series,
                                        label_in,
                                        label_out
                                        ) -> Tuple[any, any]:
    actions_no_zeros = actions[actions != 0]
    actions_labels = list(actions_no_zeros.dropna().value_counts().index)

    try:
        assert set(actions_labels).issubset({label_in, label_out})
    except AssertionError:
        try:
            assert set(actions_labels).issubset({1, -1})
            label_in = 1
            label_out = -1
        except AssertionError:
            raise Exception(f"BinPan Exception: Backtesting expected labels were not correctly specified: {actions_labels}")
    return label_in, label_out


def backtesting(df: pd.DataFrame,
                actions: pd.Series or str,
                base: float = 0,
                quote: float = 1000,
                priced_actions_col: str = 'Open',
                fee: float = 0.001,
                label_in=1,
                label_out=-1,
                suffix: str = '',
                lag_action=1,
                evaluating_quote: str = 'BUSD',
                info_dic: dict = None) -> pd.DataFrame:
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
    :param str or int label_in: A label consider as trade in trigger.
    :param str or int label_out: A label consider as trade out trigger.
    :param int lag_action: Candles needed to confirm an action from action tag. Usually one candle. Example,
       when an action like a cross of two EMA lines occur, it's needed to close that candle of the cross to confirm,
       then, nex candle can buy at open.
    :param str evaluating_quote: a Binance valid quote to evaluate operations.
    :param dict info_dic: BinPan exchange info dict to extract information about quotes and bases of symbols.
    :param str suffix: A suffix for the names of the columns.
    :return tuple: Two series with the base wallet and quote wallet funds in time.
    """

    df_ = df.copy(deep=True)

    if type(actions) == str:
        actions_ = df_[actions].copy(deep=True)
    else:
        actions_ = actions.copy(deep=True)

    # check action labels
    label_in, label_out = check_action_labels_for_backtesting(actions=actions_,
                                                              label_in=label_in,
                                                              label_out=label_out)

    base_wallet, quote_wallet = [], []

    last_action = 2314213  # any random thing
    lag = lag_action

    for idx, row in df_.iterrows():

        curr_action = actions_[idx]

        price = df_.loc[idx, priced_actions_col]

        if curr_action == label_in and last_action != 'in':
            last_action = 'in'

        elif curr_action == label_out and last_action != 'out':
            last_action = 'out'

        if last_action == 'in' and lag == 0:
            base, quote = buy_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            last_action = ''
            lag = lag_action

        elif last_action == 'in' and lag == 0:
            base, quote = sell_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            last_action = ''
            lag = lag_action
        elif last_action == 'in' or last_action == 'out':
            lag -= 1

        base_wallet.append(base)
        quote_wallet.append(quote)

    base_serie = pd.Series(base_wallet, index=df_.index, name=f"Wallet_base{suffix}")
    quote_serie = pd.Series(quote_wallet, index=df_.index, name=f"Wallet_quote{suffix}")

    if evaluating_quote:
        results_df = evaluate_wallets(df_=df_,
                                      base_serie=base_serie,
                                      quote_serie=quote_serie,
                                      evaluating_quote=evaluating_quote,
                                      info_dic=info_dic,
                                      suffix=suffix)
        return results_df

    return pd.DataFrame([base_serie, quote_serie]).T

########################
# ADVANCED BACKTESTING #
########################


def stop_loss_backtesting(df: pd.DataFrame,
                          actions_column: pd.Series or str,
                          stop_loss_column: str or pd.Series,
                          target_column: str or pd.Series,
                          fixed_stop_loss: bool,
                          base: float = 0,
                          quote: float = 1000,
                          priced_actions_col: str = 'Open',
                          fee: float = 0.001,
                          label_in=1,
                          label_out=-1,
                          suffix: str = '',
                          lag_action=1,
                          evaluating_quote: str = 'BUSD',
                          info_dic: dict = None) -> pd.DataFrame:

    df_ = df.copy(deep=True)

    if type(actions_column) == str:
        actions_data = df_[actions_column].copy(deep=True)
    else:
        actions_data = actions_column.copy(deep=True)

    if type(stop_loss_column) == str:
        stop_loss_data = df_[stop_loss_column].copy(deep=True)
    else:
        stop_loss_data = stop_loss_column.copy(deep=True)

    if type(target_column) == str:
        target_data = df_[target_column].copy(deep=True)
    else:
        target_data = target_column.copy(deep=True)

    # check action labels
    label_in, label_out = check_action_labels_for_backtesting(actions=actions_data,
                                                              label_in=label_in,
                                                              label_out=label_out)
    base_wallet, quote_wallet = [], []

    last_action = 2314213  # any random thing
    lag = lag_action
    executed_price = None
    target = None
    sl = None

    for idx, row in df_.iterrows():

        curr_action = actions_data.loc[idx]
        curr_stop_loss = stop_loss_data.loc[idx]
        curr_target = target_data.loc[idx]
        curr_low = df_['Low'].loc[idx]
        curr_high = df_['High'].loc[idx]

        price = df_.loc[idx, priced_actions_col]

        # catch actions
        if curr_action == label_in and last_action != 'in':
            last_action = 'in'
        elif curr_action == label_out and last_action != 'out':
            last_action = 'out'

        # catch start of the action usually the next candle
        if last_action == 'in' and lag == 0:
            base, quote = buy_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            last_action = ''
            lag = lag_action
            target = curr_target
            sl = curr_stop_loss

        elif last_action == 'in' and lag == 0:
            base, quote = sell_base_backtesting(row=row, price=price, base=base, quote=quote, fee=fee)
            last_action = ''
            lag = lag_action
            executed_price = price
            target = None
            sl = None

        elif last_action == 'in' or last_action == 'out':
            # if no action catch decrease lag
            lag -= 1

        if not fixed_stop_loss:
            sl = curr_stop_loss

        # check execution
        if target and sl and not executed_price:
            if curr_low <= sl:
                executed_price = sl
                target = None
                sl = None
            elif curr_high < target:
                executed_price = target
                target = None
                sl = None
            else:
                executed_price = None

        # check results
        if executed_price and not target and not sl:
            base, quote = sell_base_backtesting(row=row, price=executed_price, base=base, quote=quote, fee=fee)
            last_action = ''
            lag = lag_action

        base_wallet.append(base)
        quote_wallet.append(quote)

    base_serie = pd.Series(base_wallet, index=df_.index, name=f"Wallet_base{suffix}")
    quote_serie = pd.Series(quote_wallet, index=df_.index, name=f"Wallet_quote{suffix}")

    if evaluating_quote:
        results_df = evaluate_wallets(df_=df_,
                                      base_serie=base_serie,
                                      quote_serie=quote_serie,
                                      evaluating_quote=evaluating_quote,
                                      info_dic=info_dic,
                                      suffix=suffix)
        return results_df

    return pd.DataFrame([base_serie, quote_serie]).T
