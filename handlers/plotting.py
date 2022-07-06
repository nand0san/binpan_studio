import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from random import choice
from .logs import Logs

plot_logger = Logs(filename='./logs/plotting.log', name='plotting', info_level='INFO')

plotly_colors = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
                 "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate",
                 "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod",
                 "darkgray", "darkgrey", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
                 "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
                 "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey",
                 "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold",
                 "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
                 "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral",
                 "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgrey", "lightgreen", "lightpink",
                 "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey", "lightsteelblue",
                 "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine", "mediumblue",
                 "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen",
                 "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
                 "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered", "orchid",
                 "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru",
                 "pink", "plum", "powderblue", "purple", "red", "rosybrown", "royalblue", "rebeccapurple",
                 "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue",
                 "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal", "thistle",
                 "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"]


def set_color():
    return choice(plotly_colors)


def set_subplots(extra_rows, candles_ta_height_ratio=0.8, vertical_spacing=0.2):
    # volume is extra row
    ta_rows_heights = [(1 - candles_ta_height_ratio) / extra_rows for _ in range(extra_rows)]
    rows_heights = [candles_ta_height_ratio] + ta_rows_heights
    rows_heights = [float(i) / sum(rows_heights) for i in rows_heights]

    specs = [[{"secondary_y": False}] for _ in range(extra_rows + 1)]
    rows = 1 + extra_rows

    plot_logger.debug(rows_heights)
    plot_logger.debug(sum(rows_heights))
    plot_logger.debug(rows)
    plot_logger.debug(vertical_spacing)
    plot_logger.debug(specs)

    return make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=rows_heights,
                         vertical_spacing=vertical_spacing, specs=specs)


def set_candles(df: pd.DataFrame) -> tuple:
    c = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles')
    ax = 1
    return c, ax


# noinspection PyTypeChecker
def set_volume_series(df: pd.DataFrame, win: int = 21) -> tuple:
    # volume
    volume_green = df[df['Open'] <= df['Close']]
    volume_red = df[df['Open'] > df['Close']]

    volume_g = go.Bar(x=volume_green.index, y=volume_green['Volume'], marker_color='rgba(144,194,178,255)',
                      name='Up volume')
    volume_r = go.Bar(x=volume_red.index, y=volume_red['Volume'], marker_color='rgba(242,149,149,255)',
                      name='Down volume')
    vol_ewma = df['Volume'].ewm(span=win, min_periods=0, adjust=False, ignore_na=False).mean()
    # volume_ma = set_ta_scatter(df_, vol_ewma)
    volume_ma = go.Scatter(x=df.index, y=vol_ewma, line=dict(color='black', width=0.5), name=f'Volume ewm {win}')
    return volume_g, volume_r, volume_ma, 3


def set_ta_scatter(df: pd.DataFrame, serie: pd.Series, annotations: list = None, color='blue',
                   name='Indicator', text_position="bottom center"):
    return go.Scatter(x=df.index, y=serie, line=dict(color=color, width=0.1), name=name, mode="markers+text",
                      text=annotations, textposition=text_position)


def set_ta_line(df, serie, color='blue', name='Indicator', width=0.5, fill_color: str or bool = None):
    if fill_color:
        fill = 'tozeroy'
        fillcolor = fill_color
    else:
        fill = None
        fillcolor = None
    return go.Scatter(x=df.index, y=serie, line=dict(color=color, width=width), name=name, mode='lines', fill=fill, fillcolor=fillcolor)


def fill_missing(ll: list, length: int):
    ret = []
    cycle = 0
    for i in range(length):
        try:
            ret.append(ll[i])
        except KeyError:
            if len(ll) > 0:
                ret.append(ll[cycle])
                cycle += 1
            else:
                ret.append(f'added_{str(i).zfill(2)}')
    return ret


# noinspection PyTypeChecker
def set_arrows(annotations: pd.Series, name: str = None, tag: str = None, textposition="top center",
               mode="markers+text", marker_symbol="arrow-bar-down", marker_color='orange', marker_line_color='black',
               marker_line_width=0.5, marker_size=12):
    """Style info at https://plotly.com/python/marker-style/"""
    if not tag:
        return go.Scatter(mode=mode, x=annotations.index, y=annotations.values, text=annotations.values,
                          marker_symbol=marker_symbol, textposition=textposition, marker_line_color=marker_line_color,
                          marker_color=marker_color, marker_line_width=marker_line_width, marker_size=marker_size,
                          name=name)
    else:
        return go.Scatter(mode=mode, x=annotations.index, y=annotations.values, text=tag, marker_symbol=marker_symbol,
                          textposition=textposition, marker_line_color=marker_line_color, marker_color=marker_color,
                          marker_line_width=marker_line_width, marker_size=marker_size, name=name)


def add_traces(fig, list_of_plots: list, rows: list, cols: list):
    for i, p in enumerate(list_of_plots):
        # fig.add_trace(p, row=rows[i], col=cols[i], secondary_y=secondary_y)
        fig.append_trace(p, row=rows[i], col=cols[i])
    return fig


def set_layout_format(fig, axis_q, title, yaxis_title, width, height, range_slider):
    layout_kwargs = dict(title=title,
                         yaxis_title=yaxis_title,
                         autosize=False,
                         width=width,
                         height=height,
                         margin=dict(l=1, r=1, b=20, t=100),
                         xaxis_rangeslider_visible=range_slider,
                         xaxis_showticklabels=True)
    for i in range(axis_q):
        axis_name = 'yaxis' + str(i + 1) * (i > 0)
        layout_kwargs[axis_name] = dict(autorange=True, fixedrange=False)

    fig = fig.update_layout(layout_kwargs)
    return fig


def update_names(fig, names):
    # new_names = {'col1': 'hello', 'col2': 'hi'}
    fig.for_each_trace(lambda t: t.update(name=names[t.name],
                                          legendgroup=names[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, names[t.name])
                                          ))
    return fig


def deploy_traces(annotations, colors, markers, text_positions, mark_names, tags) -> list:
    length = len(annotations)
    if not colors:
        colors = fill_missing(['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
                               '#FF97FF', '#FECB52'], length=length)
    if not markers:
        markers = fill_missing(["arrow-bar-down", "arrow-bar-up", "arrow-bar-left", "arrow-bar-right"], length=length)
    if not text_positions:
        text_positions = ["top center" for _ in range(len(annotations))]
    if not mark_names:
        mark_names = [f"Annotation {1}" for _ in range(len(annotations))]

    annotations_traces = []  # lista de series con anotaciones
    if tags:
        for idx, an in enumerate(annotations):
            annotations_traces.append(set_arrows(annotations=an, textposition=text_positions[idx], mode="markers+text",
                                                 marker_symbol=markers[idx], marker_color=colors[idx],
                                                 name=mark_names[idx],
                                                 marker_line_color='black', marker_line_width=0.5, marker_size=15,
                                                 tag=tags[idx]))
    else:
        for idx, an in enumerate(annotations):
            annotations_traces.append(set_arrows(annotations=an, textposition=text_positions[idx], mode="markers+text",
                                                 marker_symbol=markers[idx], marker_color=colors[idx],
                                                 name=mark_names[idx],
                                                 marker_line_color='black', marker_line_width=0.5, marker_size=15))

    return annotations_traces


def candles_ta(data: pd.DataFrame,
               indicators_series: list = None,
               rows_pos: list = [],
               indicator_names: list = [],
               indicators_colors: list = [],
               indicators_color_filled: dict = None,
               width=1800,
               height=1000,
               range_slider: bool = False,
               candles_ta_height_ratio: float = 0.5,
               plot_volume: bool = True,
               title: str = 'Candlesticks, indicators, and Volume plot',
               yaxis_title: str = 'Symbol Price',
               annotations: list = None,
               markers: list = None,
               text_positions: list = None,
               annotation_colors: list = None,
               annotation_names: list = None,
               labels: list = None,
               plot_bgcolor=None):
    """
    Data needs to be a DataFrame that at least contains the columns: Open Close High Low Volume

    It plots candles and optionally volume, but can plot any list of pandas series with indicators (float values) with same index.

    Indicators will be plotted below the candles in subplots according to a row position number, counting 1 as overlay in the candles
    subplot and the rest in row subplots. Several indicators can be plotted in the same row to overlay between them and compare.

    .. note::

       Beware of zeros or values in a different scale when plotting overlapped over candles, that can break the scale of the graph.

    :param pd.DataFrame data: a DataFrame that at least contains the columns: Open Close High Low Volume
    :param list indicators_series: a list of pandas series with float values as indicators.
    :param list rows_pos: 1 means over the candles. Other numbers mean subsequent subplots under the candles.
    :param list indicator_names: Names to show in the plot. Defaults to series name.
    :param list indicators_colors: Color can be forced to anyone from the plotly colors list.

            https://community.plotly.com/t/plotly-colours-list/11730

    :param list or dict indicators_color_filled: Color can be forced to fill to zero line. Is a list of Nones for each indicator in
        indicator list or a fillcolor. For transparent colors use rgba string code to define color. Example for transparent green
        'rgba(26,150,65,0.5)' or transparent red 'rgba(204,0,0,0.5)'. It can be a dictionary with each indicator column name and fill color.
    :param int width: Plot sizing
    :param int height: Plot sizing
    :param bool range_slider: For the volume plot.
    :param float candles_ta_height_ratio: A ratio between the big candles plot and (if any) the rest of indicator subplots below.
    :param bool plot_volume: Optional to plot volume.
    :param str title: A title string.
    :param str yaxis_title: A name string.
    :param list annotations: A list of pandas series with values to plot marks or annotations overlapped in the candles plot.
    :param list markers: Ordered like the annotations list.
        Example

        .. code-block:: python

           markers = ["arrow-bar-down", "arrow-bar-up", "arrow-bar-left", "arrow-bar-right"]

    :param list text_positions: Ordered like the annotations list.
        Example

        .. code-block:: python

           text_positions = ["top center", "middle left", "top center", "bottom center", "top right", "middle left", "bottom right",
            bottom left", "top right", "top right"]

    :param list annotation_colors: Ordered like the annotations list.
        Example from default colors

        .. code-block:: python

           annotation_colors = ['cornflowerblue', 'blue', 'lightseagreen', 'green', 'cornflowerblue', 'rosybrown', 'lightseagreen',
            'black', 'orange', 'pink', 'red', 'rosybrown', 'cornflowerblue', 'blue', 'lightseagreen', 'green',
            'cornflowerblue', 'rosybrown', 'lightseagreen', 'black', 'orange', 'pink', 'red', 'rosybrown']

    :param list annotation_names: Ordered like the annotations list of names to show in legend.

    :param list labels: Ordered like the annotations list of tags to show overlapped. It defaults to price.

        Example:
        .. code-block:: python

           labels = ['buy', 'sell']

    Plot example:

        .. code-block:: python

           import binpan

           ethbtc = binpan.Symbol(symbol='ethbtc', tick_interval='1h')

           ethbtc.macd()
           binpan.handlers.plotting.candles_ta(data = ethbtc.df,
                                               indicators_series=[ethbtc.df['MACD_12_26_9_'], ethbtc.df['MACDh_12_26_9_'],
                                                    ethbtc.df['MACDs_12_26_9_']],
                                               indicators_color_filled=[False, 'rgba(26,150,65,0.5)', False],
                                               rows_pos=[2,2, 2],
                                               indicators_colors=['orange', 'green', 'skyblue'])


    .. image:: images/candles_ta_macd.png
        :width: 1000
        :alt: Candles with some indicators

    :param plot_bgcolor: Set background color.

    """
    if not indicators_color_filled:
        indicators_color_filled = {_.name: False for _ in indicators_series}
    elif type(indicators_color_filled) == list:
        indicators_color_filled = {s.name: indicators_color_filled[i] for i, s in enumerate(indicators_series)}

    plot_logger.debug(f"candles_ta indicators_color_filled: {indicators_color_filled}")

    df_plot = data.copy(deep=True)

    if not indicators_series:
        indicators_series = []

    if not indicators_colors:
        indicators_colors = ['cornflowerblue', 'blue', 'lightseagreen', 'green', 'cornflowerblue', 'rosybrown', 'lightseagreen',
                             'black', 'orange', 'pink', 'red', 'rosybrown', 'cornflowerblue', 'blue', 'lightseagreen', 'green',
                             'cornflowerblue', 'rosybrown', 'lightseagreen', 'black', 'orange', 'pink', 'red', 'rosybrown']

    if not indicator_names:
        try:
            indicator_names = [i.name for i in indicators_series]
        except Exception:
            indicator_names = [f'Indicator {i}' for i in range(len(indicators_series))]

    if not indicators_color_filled:
        indicators_color_filled = [False for _ in indicators_series]

    if plot_volume:
        extra_rows = len(set(rows_pos)) + 1
    else:
        extra_rows = len(set(rows_pos))

    fig = set_subplots(extra_rows=extra_rows, candles_ta_height_ratio=candles_ta_height_ratio, vertical_spacing=0.02)

    # limit
    axes = 0
    candles_plot, ax = set_candles(df_plot)
    axes += ax

    # volume
    if plot_volume:
        volume_g, volume_r, volume_ma, ax = set_volume_series(df_plot)
        axes += ax
        rows = [1, 2, 2, 2]
        rows_pos = [i + 1 if i != 1 else i for i in rows_pos]
        traces = [candles_plot, volume_g, volume_r, volume_ma]
    else:
        traces = [candles_plot]
        rows = [1]

    cols = [1 for _ in range(len(rows))]

    # technical analysis indicators
    tas = []

    plot_logger.debug(f"{indicators_colors}")
    plot_logger.debug(f"{indicators_color_filled}")

    for i, indicator in enumerate(indicators_series):
        tas.append(set_ta_line(df=df_plot, serie=indicator, color=indicators_colors[i], name=indicator_names[i],
                               width=1, fill_color=list(indicators_color_filled.values())[i]))
        axes += 1

    cols = cols + [1 for _ in range(len(tas))]
    rows = rows + [i for i in rows_pos]
    traces = traces + tas

    # anotaciones, siempre van en la primera fila, la de las velas
    if annotations:
        annotations_traces = deploy_traces(annotations=annotations, colors=annotation_colors, markers=markers,
                                           text_positions=text_positions, mark_names=annotation_names, tags=labels)
        rows += [1 for _ in range(len(annotations))]
        cols += [1 for _ in range(len(annotations))]
        traces += annotations_traces

    fig = add_traces(fig, traces, rows=rows, cols=cols)
    fig = set_layout_format(fig, axes, title, yaxis_title, width, height, range_slider)

    if plot_bgcolor:
        fig.update_layout(plot_bgcolor=plot_bgcolor)

    fig.show()


def candles_tagged(data: pd.DataFrame, width=1800, height=1000, candles_ta_height_ratio=0.5,
                   plot_volume=True,
                   title: str = 'Candlesticks Strategy Plot',
                   yaxis_title: str = 'Symbol Price',
                   on_candles_indicator: list = [],
                   priced_actions_col='priced_actions',
                   actions_col: str = None,
                   indicators_series: list = [],
                   indicator_names: list = [],
                   indicators_colors: list = [],
                   fill_control: dict or list = None,
                   rows_pos: list = [],
                   labels: list = [],
                   default_price_for_actions='Close',
                   plot_bgcolor=None):
    """

    This is a shortcut from candles_ta. It defaults many inputs to better Jupyter Notebook usage.

    Data needs to be a DataFrame that at least contains the columns: Open Close High Low Volume

    It plots candles and optionally volume, but can plot any list of pandas series with indicators (float values) with same index.

    Indicators will be plotted below the candles in subplots according to a row position number, counting 1 as overlay in the candles
    subplot and the rest in row subplots. Several indicators can be plotted in the same row to overlay between them and compare.

    .. note::

        Beware of zeros or values in a different scale when plotting overlapped over candles, that can break the scale of the graph.

    Plot example:

        .. image:: images/plot_tagged.png
            :width: 1000
            :alt: Candles with some indicators


    :param pd.DataFrame data: a DataFrame that at least contains the columns: Open Close High Low Volume
    :param list indicators_series: a list of pandas series with float values as indicators.
    :param list rows_pos: 1 means over the candles. Other numbers mean subsequent subplots under the candles.
    :param list indicator_names: Names to show in the plot. Defaults to series name.
    :param list indicators_colors: Color can be forced to anyone from the plotly colors list:

         https://community.plotly.com/t/plotly-colours-list/11730

    :param dict or list fill_control: A dictionary with color to fill or False bool for each indicator. Is the color to the zero line for
        the indicator plot. If a list passed, it iterates to assign each item in the list with the same index item in the indicators list.
    :param int width: Plot sizing
    :param int height: Plot sizing
    :param float candles_ta_height_ratio: A ratio between the big candles plot and (if any) the rest of indicator subplots below.
    :param bool plot_volume: Optional to plot volume.
    :param str title: A title string.
    :param str yaxis_title: A name string.
    :param actions_col: A column name of the column with string tags like buy, sell, etc. This is for plotting marks over candles.
    :param priced_actions_col: The name of the column containing value of action to position over candles.
    :param on_candles_indicator: A list of pandas series with values to plot over the candles, not in a subplot. Example: SMA.
    :param list labels: Ordered like the annotations list of tags to show overlapped. It defaults to price. Example:

        .. code-block:: python

          labels = ['buy', 'sell']

    :param default_price_for_actions: Column to pick prices for actions, because actions will be labeled, in example, actions values like
        buy or sell with an arrow will be positioned from the close price if not exists a more precise value for the action.

    Example:

    .. code-block:: python

       import binpan

       btcusdt = binpan.Symbol(symbol='btcusdt',
                        tick_interval='15m',
                        time_zone='Europe/Madrid',
                        end_time='2021-10-31 03:00:00')

       btcusdt.sma(21)

       df = binpan.handlers.strategies.random_strategy(data=btcusdt.df, 10, 10)

       df.actions.value_counts()

       -       980
       sell     10
       buy      10
       Name: actions, dtype: int64

       binpan.handlers.plotting.candles_tagged(data=df,
                                               plot_volume=False,
                                               indicators_series=[df.SMA_21],
                                               candles_ta_height_ratio=0.8,
                                               actions_col='actions',
                                               labels=['buy', 'sell'])


    .. image:: images/plot_tagged_example_02.png
       :width: 1000

    :param plot_bgcolor: Set background color.

    """

    if type(fill_control) == list:
        fill_control = {s.name: fill_control[i] for i, s in enumerate(indicators_series)}

    markers = ["arrow-bar-up", "arrow-bar-down"]
    annotation_colors = ['green', 'red']
    annotations_names = ['Buy', 'Sell']
    rows_pos_final = []

    if on_candles_indicator:
        rows_pos_final = [1 for _ in range(len(on_candles_indicator))]

    if indicators_series and not rows_pos:
        rows_pos_final += [i + 2 for i in range(len(indicators_series))]
    else:
        rows_pos_final += rows_pos + [i + 2 for i in range(len(indicators_series) - len(rows_pos))]

    indicators_series = on_candles_indicator + indicators_series

    if priced_actions_col in data.columns:
        annotations_values = [data[data[actions_col] == 'buy'][priced_actions_col],
                              data[data[actions_col] == 'sell'][priced_actions_col]]
    elif actions_col:
        annotations_values = [data[data[actions_col] == 'buy'][default_price_for_actions],
                              data[data[actions_col] == 'sell'][default_price_for_actions]]
    else:
        annotations_values = None
    if not indicator_names:
        try:
            indicator_names = [i.name for i in indicators_series]
        except Exception:
            pass

    candles_ta(data,
               width=width,
               height=height,
               range_slider=False,
               candles_ta_height_ratio=candles_ta_height_ratio,
               plot_volume=plot_volume,
               title=title,
               yaxis_title=yaxis_title,
               annotations=annotations_values,
               markers=markers,
               rows_pos=rows_pos_final,
               annotation_colors=annotation_colors,
               annotation_names=annotations_names,
               indicators_series=indicators_series,
               indicator_names=indicator_names,
               indicators_colors=indicators_colors,
               indicators_color_filled=fill_control,
               labels=labels,
               plot_bgcolor=plot_bgcolor)


def plot_trade_size(data: pd.DataFrame, max_size=60, height=1000, logarithmic=False, title=f"Trade Size"):
    """
    Plots scatter plot from trades quantity and trades sizes. Marks are size scaled to the max size. Marks are semi transparent and colored
    using Maker buyer or Taker buyer discrete colors. Usually red and blue.

    Can let you see where are the big sized trades done and the taker or maker buyer side.

    :param pd.DataFrame data: A BinPans trades dataframe.
    :param int max_size: Size of the marks for the biggest quantity sized trades.
    :param int height: Plot sizing.
    :param bool logarithmic: Y axis in a logarithmic scale.
    :param str title: Title string.

    Example:
        .. code-block:: python

           import binpan
           lunc = binpan.Symbol(symbol='luncbusd',
                                tick_interval='5m',
                                limit = 100,
                                time_zone = 'Europe/Madrid',
                                time_index = True,
                                closed = True)
           lunc.get_trades()
           binpan.handlers.plotting.plot_trade_size(data = lunc.trades, logarithmic=True)

        .. image:: images/plot_trades_size_log.png
           :width: 1000
           :alt: Trades size

    """
    data['Buyer was maker'].replace({True: 'Maker buyer', False: 'Taker buyer'}, inplace=True)
    fig = px.scatter(x=data.index, y=data['Price'], color=data['Buyer was maker'], size=data['Quantity'],
                     title=title,
                     height=height, size_max=max_size, log_y=logarithmic)
    fig.show()


def normalize(max_value, min_value, data: list):
    return [(i / sum(data)) * max_value + min_value for i in data]


def plot_pie(serie: pd.Series,
             categories: int = 15,
             title=f"Size trade categories",
             logarithmic=False):
    """
    Plots a pie chart from a column. Useful to see size ranges in trades, but can be used in any way.

    :param pd.Series serie: pandas serie with numeric values or strings.
    :param int categories: Category count to divide chart.
    :param str title: String title.
    :param bool logarithmic: If logarithmic is selected as true, the sizes of each interval will be distributed in logarithmic steps from
     the smallest to the largest, that is, the smallest values will be divided into smaller groups that will increase exponentially in size.

    Example:

        .. code-block:: python

           import binpan

           lunc = binpan.Symbol(symbol='luncbusd',
                                tick_interval='5m',
                                limit = 100,
                                time_zone = 'Europe/Madrid',
                                time_index = True,
                                closed = True)

           lunc.get_trades()

           binpan.handlers.plotting.plot_pie(serie = lunc.trades['Quantity'], logarithmic=True)

        .. image:: images/plot_pie_log.png
           :width: 1000
           :alt: Trades size
    """
    ma_original = serie.max()
    mi_original = serie.min()
    integer_size = len(str(ma_original).split('.')[0])
    plot_logger.debug(f"integer_size: {integer_size}, max:{ma_original} min:{mi_original}")

    ma = ma_original / 10 ** (integer_size - 1)
    mi = mi_original / 10 ** (integer_size - 1)

    if logarithmic:
        category_steps = 10. ** np.linspace(mi, ma, categories)
        plot_logger.debug(f"category_steps: {category_steps}")
        category_steps = normalize(max_value=ma, min_value=mi, data=category_steps)
        plot_logger.debug(f"category_steps: {category_steps}")

        spread = [mi_original] + normalize(max_value=ma_original, min_value=mi_original, data=category_steps) + [ma_original]
        plot_logger.debug(f"spread: {spread}")
        plot_logger.debug(f"order: {spread}")
    else:
        step = (ma_original - mi_original) / categories
        spread = np.arange(mi_original, ma_original, step)

    # orders = {serie.name: spread}

    pie = serie.groupby(pd.cut(serie, spread)).count()
    names = [str(i) for i in pie.index]

    fig = px.pie(pie,
                 values=serie.name,
                 names=names,
                 color_discrete_sequence=px.colors.sequential.RdBu,
                 title=title,
                 hover_name=serie.name)
    # category_orders=orders)
    fig.show()


def plot_scatter(df: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 symbol: str = None,
                 color: str = None,
                 marginal: bool = True,
                 title: str = None,
                 height: int = 1000,
                 **kwargs):
    """
    Plot scatter plots with a column of values in X axis and other in Y axis.

    :param pd.DataFrame df: A Dataframe.
    :param str x_col: Name of column with X axis data.
    :param str y_col: Name of column with Y axis data.
    :param str symbol: Name of column with values (discrete or not) to apply a symbol each.
    :param str color: Name of column with values (discrete or not) to apply a color each.
    :param bool marginal: Lateral auxiliar plots.
    :param str title: A title string.
    :param height: Plot sizing.
    :param kwargs: Optional plotly kwargs.

    Example:

    .. code-block::

       lunc = binpan.Symbol(symbol='luncbusd',
                            tick_interval='5m',
                            limit = 100,
                            time_zone = 'Europe/Madrid',
                            time_index = True,
                            closed = True)

       binpan.handlers.plotting.plot_scatter(df = lunc.df,
                                             x_col='Close',
                                             y_col='Volume',
                                             color='Trades',
                                             symbol='Close',
                                             title='Scatter plot for LUNCBUSD Close price in X and Volume in Y'
                                             )

    .. image:: images/scatter_example.png
       :width: 1000
       :alt: Scatter plot example

    """
    if marginal:
        fig = px.scatter(df,
                         x=x_col,
                         y=y_col,
                         symbol=symbol,
                         color=color,
                         title=title,
                         marginal_x="histogram",
                         marginal_y="rug",
                         height=height,
                         **kwargs)
    else:
        fig = px.scatter(df,
                         x=x_col,
                         y=y_col,
                         symbol=symbol,
                         color=color,
                         title=title,
                         height=height,
                         **kwargs)
    fig.show()


def plot_hists_vs(x0: pd.Series,
                  x1: pd.Series,
                  x0_name: str = None,
                  x1_name: str = None,
                  bins: int = 50,
                  hist_funct: str = 'sum',
                  height: int = 900,
                  title: str = None,
                  **kwargs_update_layout):
    """
    Plots two histograms with same x scale to campare distributions of values.

    :param pd.Series x0: A pandas series.
    :param pd.Series  x1: A pandas series.
    :param str x0_name: Name for the legend
    :param str x1_name: Name for the legend
    :param int bins: Number of bins or bars to show.
    :param str hist_funct: A function to apply to data. It can be 'sum', 'count', 'average', etc...

        More details in: https://plotly.com/python/histograms/#histograms-with-gohistogram

    :param int height: Plot sizing.
    :param str title: Plot title.
    :param kwargs_update_layout: Plotly update layout options.


    Example:

    .. code-block::

       lunc = binpan.Symbol(symbol='luncbusd',
                            tick_interval='5m',
                            limit = 100,
                            time_zone = 'Europe/Madrid',
                            time_index = True,
                            closed = True)

       binpan.handlers.plotting.plot_hists_vs(x0=lunc.df['High'],
                                              x1=lunc.df['Low'],
                                              bins=50,
                                              hist_funct='count',
                                              title='High and Low prices distribution.')

    .. image:: images/hist_vs_dist.png
       :width: 1000
       :alt: Histogram plot example


    """
    if not x0_name:
        x0_name = x0.name
    if not x1_name:
        x1_name = x1.name

    fig = go.Figure()

    start = min(x0.min(), x1.min())
    end = max(x0.max(), x1.max())

    fig.add_trace(go.Histogram(x=x0,
                               histfunc=hist_funct,
                               name=x0_name,
                               xbins=dict(
                                   start=start,
                                   end=end,
                                   size=(x0.max() - x0.min()) / bins)))

    fig.add_trace(go.Histogram(x=x1,
                               histfunc=hist_funct,
                               name=x1_name,
                               xbins=dict(
                                   start=start,
                                   end=end,
                                   size=(x0.max() - x0.min()) / bins)))

    fig.update_layout(
        bargap=0.3,
        title=title,
        xaxis_title_text=f'{x0_name} vs {x1_name} size',
        yaxis_title_text=f'{x0_name} vs {x1_name} {hist_funct}',
        bargroupgap=0.1,
        height=height,
        **kwargs_update_layout)

    fig.update_traces(opacity=0.75)
    fig.show()


def orderbook_depth(df: pd.DataFrame,
                    accumulated=True,
                    title='Depth orderbook plot',
                    height=500,
                    plot_y="Quantity",
                    **kwargs):
    """
    Plots orderbook from a BinPan orderbook dataframe.

    :param pd.DAtaFrame df: BinPan orderbook dataframe.
    :param bool accumulated: If true, applies cumsum to asks and bids.
    :param str title: A title string.
    :param int height: Plot sizing.
    :param str plot_y: Column name with y axis data. Defaults to Quantity.
    :param kwargs: Plotly kwargs.

    Example:

    .. code-block::

        lunc = binpan.Symbol(symbol='luncbusd',
                            tick_interval='5m',
                            limit = 100,
                            time_zone = 'Europe/Madrid',
                            time_index = True,
                            closed = True)

        lunc.get_orderbook()

    .. image:: images/plot_orderbook.png
       :width: 1000
       :alt: Plot example

    """
    ob = df.copy(deep=True)
    if accumulated:
        c_asks = ob[ob['Side'] == 'ask']['Quantity'][::-1].cumsum()
        c_bids = ob[ob['Side'] == 'bid']['Quantity'].cumsum()
        cumulated = pd.concat([c_asks, c_bids])
        ob.loc[:, 'Accumulated Quantity'] = cumulated
        plot_y = 'Accumulated Quantity'

    fig = px.line(ob, x="Price", y=plot_y, color='Side', height=height, title=title, **kwargs)
    fig.show()
