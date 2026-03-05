"""

Strategy and backtesting mixin for Symbol class.

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from handlers.tags import (tag_column_to_strategy_group, backtesting as run_backtesting,
                           backtesting_short, tag_comparison, tag_cross, merge_series,
                           clean_in_out as clean_in_out_func)
from handlers.indicators import ffill_indicator, shift_indicator


class StrategyMixin:
    """Mixin that adds tagging, strategy and backtesting methods to Symbol."""

    ###############
    # Backtesting #
    ###############

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
                    label_in=1, label_out=-1,
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

        if type(actions_col) == int:
            actions = self.df.iloc[:, actions_col]
        else:
            actions = self.df[actions_col]

        if suffix:
            suffix = '_' + suffix

        if not short:
            wallet_df = run_backtesting(df=self.df, actions_column=actions, target_column=target_column, stop_loss_column=stop_loss_column,
                                    entry_filter_column=entry_filter_column, priced_actions_col=priced_actions_col,
                                    fixed_target=fixed_target,
                                    fixed_stop_loss=fixed_stop_loss, base=base, quote=quote, fee=fee, label_in=label_in,
                                    label_out=label_out, suffix=suffix,
                                    evaluating_quote=evaluating_quote, info_dic=self.info_dic)
        else:
            wallet_df = backtesting_short(df=self.df, actions_column=actions, target_column=target_column,
                                          stop_loss_column=stop_loss_column, entry_filter_column=entry_filter_column,
                                          priced_actions_col=priced_actions_col,
                                          fixed_target=fixed_target, fixed_stop_loss=fixed_stop_loss, base=base, quote=quote, fee=fee,
                                          label_in=label_in,
                                          label_out=label_out, suffix=suffix, evaluating_quote=evaluating_quote, info_dic=self.info_dic)

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

    def roi(self, column: str = None) -> float:
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

    def profit_hour(self, column: str = None) -> float:
        """
        It returns win or loos quantity per hour. Just compares first and last value. Expected datetime index. If not column passed, it
        will search for an Evaluation column.

        :param str column: A column in the BinPan's DataFrame with values to check profit with expected datetime index.
        :return float: Resulting return of inversion.
        """
        if not column:
            column = [i for i in self.df.columns if i.startswith('Eval')]
            if not column:
                column = "Close"
            else:
                column = column[-1]
            print(f"Auto selected column {column}")

        my_column = self.df[column].copy(deep=True)
        my_column.dropna(inplace=True)

        first = my_column.iloc[0]
        last = my_column.iloc[-1]

        profit = last - first
        ms = self.df['Close timestamp'].dropna().iloc[-1] - self.df['Open timestamp'].dropna().iloc[0]
        hours = ms / (1000 * 60 * 60)

        print(f"Total profit for {column}: {profit} with ratio {profit / hours} per hour.")

        return profit / hours

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
            inplace=True, suffix: str = '',
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

           import binpan

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

        compared = tag_comparison(serie_a=data_a, serie_b=data_b, **{relation: True}, match_tag=match_tag, mismatch_tag=mismatch_tag)

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
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return compared

    def cross(self,
              slow: str or int or float or pd.Series,
              fast: str or int or pd.Series = 'Close',
              cross_over_tag: str or int = 1,
              cross_below_tag: str or int = -1, echo=0,
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

           import binpan

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

        cross = tag_cross(serie_a=data_a, serie_b=data_b, echo=echo, cross_over_tag=cross_over_tag, cross_below_tag=cross_below_tag,
                          name=column_name, non_zeros=non_zeros)

        if inplace and self.is_new(cross):
            self.row_counter += 1
            self.set_plot_color(indicator_column=column_name, color=color)
            self.set_plot_color_fill(indicator_column=column_name, color_fill=None)
            self.set_plot_row(indicator_column=str(column_name), row_position=self.row_counter)  # overlaps are one
            self.df.loc[:, column_name] = cross

        if strategy_group:
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return cross

    def shift(self,
              column: str or int or pd.Series, window=1,
              strategy_group: str = '',
              inplace=True, suffix: str = '',
              color: str or int = 'grey'):
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
        shift = shift_indicator(serie=data_a, window=window)
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
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)

        return shift

    def merge_columns(self,
                      main_column: str or int or pd.Series,
                      other_column: str or int or pd.Series,
                      sign_other: dict = None,
                      strategy_group: str = '',
                      inplace=True,
                      suffix: str = '',
                      color: str or int = 'grey'):
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
        if not sign_other:
            sign_other = {1: -1}
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

        merged = merge_series(predominant=data_a, other=data_b)
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
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return merged

    def clean_in_out(self,
                     column: str or int or pd.Series,
                     in_tag=1,
                     out_tag=-1,
                     strategy_group: str = '',
                     inplace=True, suffix: str = '',
                     color: str or int = 'grey'):
        """
        It cleans a serie with in and out tags by eliminating in streaks and out streaks.

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
            data_a = self.df.iloc[:, column].copy(deep=True)
        else:
            data_a = column.copy(deep=True)

        clean = clean_in_out_func(serie=data_a, in_tag=in_tag, out_tag=out_tag)
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
            self.strategy_groups = tag_column_to_strategy_group(column=column_name, group=strategy_group,
                                                                strategy_groups=self.strategy_groups)
        return clean

    def set_strategy_groups(self, column: str, group: str, strategy_groups: dict = None):
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
            self.strategy_groups = tag_column_to_strategy_group(column=column, group=group, strategy_groups=strategy_groups)
        return self.strategy_groups

    def get_strategy_columns(self) -> list:
        """
        Returns column names starting with "Strategy".

        :return dict: Updated strategy groups of columns.
        """
        return [i for i in self.df.columns if i.lower().startswith('strategy')]

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

    def ffill_window(self,
                     column: str or int or pd.Series,
                     window: int = 1,
                     inplace=True,
                     replace=False,
                     suffix: str = '',
                     color: str or int = 'blue'):
        """
        It forward fills a value through nans a window ahead.

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

        my_ffill = ffill_indicator(serie=serie, window=window)

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
