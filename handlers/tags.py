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
    print(locals())
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
    ret.loc[ret==False] = mismatch_tag
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
    It tags values of a serie compared to other serie by methos gt,ge,eq,le,lt condition.

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
    print(locals())
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
              strict_mode: bool = True,
              match_over_tag='buy',
              match_below_tag='sell',
              name='Cross') -> pd.Series:
    """
    It tags points where serie_a crosses over serie_b or optionally below.

    :param pd.Series serie_a: A numeric serie.
    :param pd.Series serie_b: A numeric serie.
    :param int echo: It tags a fixed amount of candles forward the crossed point not including cross candle.
    :param bool strict_mode: Forces strictly greater and strictly lower values comparison.
    :param int or str match_over_tag: Value to tag over cross.
    :param int or str match_below_tag: Value to tag below cross.
    :param str name: Name for the resulting serie.
    :return pd.Series: A serie with tags as values.
    """
    serie_a_ = serie_a.copy(deep=True)
    serie_b_ = serie_b.copy(deep=True)
    serie_a_shift = serie_a_.shift()
    serie_b_shift = serie_b_.shift()

    ret = pd.Series(index=serie_a.index)

    if strict_mode:
        ret_over = ((serie_a_.gt(serie_b_)) & (serie_a_shift.le(serie_b_shift)))
        ret_below = ((serie_a_.lt(serie_b_)) & (serie_a_shift.ge(serie_b_shift)))

    else:
        ret_over = ((serie_a_.ge(serie_b_)) & (serie_a_shift.lt(serie_b_shift)))
        ret_below = ((serie_a_.le(serie_b_)) & (serie_a_shift.gt(serie_b_shift)))

    ret.loc[ret_over] = match_over_tag
    ret.loc[ret_below] = match_below_tag
    ret.name = name
    if echo:
        return ret.ffill(limit=echo)
    else:
        return ret
