import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from random import choice
from .logs import Logs
from .exceptions import BinPanException

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

    plot_logger.debug(f"rows_heights: {rows_heights}")
    plot_logger.debug(f"sum(rows_heights): {sum(rows_heights)}")
    plot_logger.debug(f"rows: {rows}")
    plot_logger.debug(f"vertical_spacing: {vertical_spacing}")
    plot_logger.debug(f"specs: {specs}")

    return make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=rows_heights,
                         vertical_spacing=vertical_spacing, specs=specs)


def set_candles(df: pd.DataFrame) -> tuple:
    candles_plot = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles')
    ax = 1
    return candles_plot, ax


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


def set_ta_scatter(df: pd.DataFrame,
                   serie: pd.Series,
                   annotations: list = None,
                   color='blue',
                   name='Indicator',
                   text_position="bottom center"):

    return go.Scatter(x=df.index,
                      y=serie,
                      line=dict(color=color, width=0.1),
                      name=name,
                      mode="markers+text",
                      text=annotations,
                      textposition=text_position)


def set_ta_line(df_index: pd.DataFrame.index,
                serie: pd.Series,
                color='blue',
                name='Indicator',
                width=0.5,
                fill_color: str or bool = None,
                fill_mode: str = 'none',
                yaxis: str = 'y',
                show_legend=True):
    my_locals = {k: v for k, v in locals().items() if k != 'df_index' and k != 'serie'}
    plot_logger.debug(f"set_ta_line: {my_locals}")

    if fill_mode:
        fillcolor = fill_color
    else:
        fillcolor = None

    return go.Scatter(x=df_index,
                      y=serie,
                      line=dict(color=color, width=width),
                      name=name,
                      mode='lines',
                      fill=fill_mode,
                      fillcolor=fillcolor,
                      yaxis=yaxis,
                      showlegend=show_legend)


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


def set_layout_format(fig, axis_q: int,
                      title: str,
                      yaxis_title: str,
                      width: int,
                      height: int,
                      range_slider: bool):
    layout_kwargs = dict(title=title,
                         yaxis_title=yaxis_title,
                         autosize=False,
                         width=width,
                         height=height,
                         margin=dict(l=1, r=1, b=20, t=100),
                         xaxis_rangeslider_visible=range_slider,
                         xaxis_showticklabels=True)
    # renaming axis names
    for i in range(axis_q):
        axis_name = 'yaxis' + str(i + 1) * (i > 0)
        layout_kwargs[axis_name] = dict(autorange=True, fixedrange=False)  # los subplots pintan bien los datos aunque se expanda el index

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


###################
# market plotting #
###################

def candles_ta(data: pd.DataFrame,
               indicators_series: list = None,
               rows_pos: list = [],
               indicator_names: list = [],
               indicators_colors: list = [],
               indicators_color_filled: dict = None,
               indicators_filled_mode: dict = None,
               axis_groups: dict = {},
               plot_splitted_serie_couple: dict = {},
               width=1800,
               height=1000,
               range_slider: bool = False,
               candles_ta_height_ratio: float = 0.5,
               plot_volume: bool = True,
               title: str = 'Candlesticks, indicators, and Volume plot',
               yaxis_title: str = 'Symbol Price',
               annotation_values: list = None,
               markers: list = None,
               text_positions: list = None,
               annotation_colors: list = None,
               annotation_legend_names: list = None,
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
    :param dict indicators_filled_mode: A dict with filled areas for plotting.
    :param dict axis_groups: A dict with named groups for indicators, useful for plotting filled areas using tonexty fill mode.
    :param dict plot_splitted_serie_couple: A dict with splitted data for multiple colours when filling areas using tonexty.
    :param int width: Plot sizing
    :param int height: Plot sizing
    :param bool range_slider: For the volume plot.
    :param float candles_ta_height_ratio: A ratio between the big candles plot and (if any) the rest of indicator subplots below.
    :param bool plot_volume: Optional to plot volume.
    :param str title: A title string.
    :param str yaxis_title: A name string.
    :param list annotation_values: A list of pandas series with values to plot marks or annotations overlapped in the candles plot.
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

    :param list annotation_legend_names: Ordered like the annotations list of names to show in legend.

    :param list labels: Ordered like the annotations list of tags to plot overlapped. It defaults to price value if omitted.

        Example:
        .. code-block:: python

           labels = ['buy', 'sell']

    Plot example:

        .. code-block:: python

            from binpan import binpan

            ethbtc = binpan.Symbol(symbol='ethbtc', tick_interval='1h')

            ethbtc.macd(fast=12, slow=26, smooth=9)
            print(ethbtc.df)

            binpan.handlers.plotting.candles_ta(data=ethbtc.df,
                                                indicators_series=[ethbtc.df['MACD_12_26_9'],
                                                                   ethbtc.df['MACDh_12_26_9'],
                                                                   ethbtc.df['MACDs_12_26_9']],
                                                indicators_color_filled=[False, 'rgba(26,150,65,0.5)', False],
                                                rows_pos=[2, 2, 2],
                                                indicators_colors=['orange', 'green', 'skyblue'])


    .. image:: images/candles_ta_macd.png
        :width: 1000
        :alt: Candles with some indicators

    :param plot_bgcolor: Set background color.

    """
    if not indicators_color_filled:
        indicators_color_filled = {i.name: None for i in indicators_series}
    elif type(indicators_color_filled) == list:
        indicators_color_filled = {s.name: indicators_color_filled[i] for i, s in enumerate(indicators_series)}
    plot_logger.debug(f"candles_ta indicators_color_filled: {indicators_color_filled}")

    if not indicators_filled_mode:
        indicators_filled_mode = {i.name: None for i in indicators_series}
    elif type(indicators_filled_mode) == list:
        indicators_filled_mode = {s.name: indicators_filled_mode[i] for i, s in enumerate(indicators_series)}
    plot_logger.debug(f"candles_ta indicators_filled_mode: {indicators_filled_mode}")

    df_plot = data.copy(deep=True)

    if not indicators_series:
        indicators_series = []

    if not indicators_colors:
        indicators_colors = [choice(plotly_colors) for _ in range(len(indicators_series))]

    if not indicator_names:
        try:
            indicator_names = [i.name for i in indicators_series]
        except Exception:
            indicator_names = [f'Indicator {i}' for i in range(len(indicators_series))]

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
        pre_rows = 4
        rows_pos = [i + 1 if i != 1 else i for i in rows_pos]
        traces = [candles_plot, volume_g, volume_r, volume_ma]
    else:
        traces = [candles_plot]
        rows = [1]
        pre_rows = 1

    rows = rows + [i for i in rows_pos]
    cols = [1 for _ in range(len(rows))]

    # technical analysis indicators
    tas = []

    y_axis_idx = [f"y{i}" for i in rows]

    plot_logger.debug(f"----------------------------------------------------------------------")
    plot_logger.debug(f"indicators_colors: {indicators_colors} len: {len(indicators_colors)}")
    plot_logger.debug(f"indicators_color_filled: {indicators_color_filled} len: {len(indicators_color_filled)}")
    plot_logger.debug(f"indicators_filled_mode: {indicators_filled_mode} len: {len(indicators_filled_mode)}")
    plot_logger.debug(f"rows: {rows} len: {len(rows)}")
    plot_logger.debug(f"indicators_series: {len(indicators_series)} len: {len(indicators_series)}")
    plot_logger.debug(f"y_axis_idx: {y_axis_idx} len: {len(y_axis_idx)}")
    plot_logger.debug(f"axis_groups: {axis_groups} len: {len(axis_groups)}")
    plot_logger.debug(f"plot_splitted_serie_couple: {plot_splitted_serie_couple} len: {len(plot_splitted_serie_couple)}")
    plot_logger.debug(f"----------------------------------------------------------------------")

    # first get tas with cloud colors "tonexty"
    pre_cached = 0
    for i, indicator in enumerate(indicators_series):
        plot_logger.debug(f"Loop plotting: indicator.name={indicator.name}")
        pre_i = i + pre_rows + pre_cached

        if indicator.name in indicators_filled_mode.keys():
            my_fill_mode = indicators_filled_mode[indicator.name]
        else:
            my_fill_mode = None

        if indicator.name in indicators_color_filled.keys():
            my_fill_color = indicators_color_filled[indicator.name]
        else:
            my_fill_color = None

        if indicator.name in axis_groups.keys():
            my_axis = axis_groups[indicator.name]
        else:
            my_axis = y_axis_idx[pre_i]

        # my_axis_from_cache_100 = f"y1{my_axis[1:]}"

        if indicator_names[i] in plot_splitted_serie_couple.keys():
            plot_logger.debug(f"indicator splitted: {indicator_names[i]}")
            # serie_up, split_up, serie_down, split_down, color_up, color_down = plot_splitted_serie_couple[indicator_names[i]]
            # plot_logger.debug(f"serie_up, split_up, serie_down, split_down, color_up, color_down = {serie_up, split_up, serie_down, split_down, color_up, color_down}")
            indicator_column_up, indicator_column_down, splitted_dfs, color_up, color_down = plot_splitted_serie_couple[indicator_names[i]]
            plot_logger.debug(f"indicator_column_up, indicator_column_down, splitted_dfs,color_up, color_down = "
                              f"{indicator_column_up, indicator_column_down, splitted_dfs,color_up, color_down}")

            tas.append(set_ta_line(df_index=df_plot.index,  # linea para delimitación
                                   serie=indicator,
                                   color=indicators_colors[i],
                                   name=indicator_names[i],
                                   width=1,
                                   fill_mode='none',
                                   fill_color=None,
                                   yaxis=my_axis))
            # cambio de función

            def fill_area(label,
                          up_color='rgba(35, 152, 33, 0.5)',
                          down_color='rgba(245, 63, 39, 0.5)'):
                if label >= 1:
                    return up_color
                else:
                    return down_color

            for splitted_df in splitted_dfs:

                tas.append(set_ta_line(df_index=splitted_df.index,
                                       serie=splitted_df[indicator_column_up],
                                       color=indicators_colors[i],
                                       name=indicator_column_up,
                                       width=0.01,
                                       fill_mode='none',
                                       fill_color=None,
                                       yaxis=my_axis,
                                       show_legend=False))

                tas.append(set_ta_line(df_index=splitted_df.index,
                                       serie=splitted_df[indicator_column_down],
                                       color=indicators_colors[i],
                                       name=indicator_column_down,
                                       width=0.01,
                                       fill_mode='tonexty',
                                       fill_color=fill_area(splitted_df['label'].iloc[0]),
                                       yaxis=my_axis,
                                       show_legend=False))

                rows = rows[:pre_i] + [rows[pre_i], rows[pre_i]] + rows[pre_i:]
                pre_cached += 2
                y_axis_idx = y_axis_idx[:pre_i] + [my_axis, my_axis] + y_axis_idx[pre_i:]

            plot_logger.debug(f"rows_updated_by_split: {rows} len: {len(rows)}")
            plot_logger.debug(f"y_axis_idx_updated_by_split: {y_axis_idx} len: {len(y_axis_idx)}")
        else:
            plot_logger.debug(f"indicator_name: {indicator_names[i]}: row: {rows[pre_i]} axis: {my_axis}")

            tas.append(set_ta_line(df_index=df_plot.index,
                                   serie=indicator,
                                   color=indicators_colors[i],
                                   name=indicator_names[i],
                                   width=1,
                                   fill_mode=my_fill_mode,
                                   fill_color=my_fill_color,
                                   yaxis=my_axis))
        axes += 1

    cols = cols + [1 for _ in range(len(tas))]
    traces = traces + tas

    # anotaciones, siempre van en la primera fila, la de las velas
    if annotation_values:
        annotations_traces = deploy_traces(annotations=annotation_values, colors=annotation_colors, markers=markers,
                                           text_positions=text_positions, mark_names=annotation_legend_names, tags=labels)
        rows += [1 for _ in range(len(annotation_values))]
        cols += [1 for _ in range(len(annotation_values))]
        traces += annotations_traces

    # use different traces for cloud indicators
    fig = add_traces(fig=fig,
                     list_of_plots=traces,
                     rows=rows,
                     cols=cols)

    fig = set_layout_format(fig=fig,
                            axis_q=axes,
                            title=title,
                            yaxis_title=yaxis_title,
                            width=width,
                            height=height,
                            range_slider=range_slider)

    if plot_bgcolor:
        fig.update_layout(plot_bgcolor=plot_bgcolor)

    fig.show()


def candles_tagged(data: pd.DataFrame,
                   width=1800,
                   height=1000,
                   candles_ta_height_ratio=0.5,
                   plot_volume=True,
                   title: str = 'Candlesticks Strategy Plot',
                   yaxis_title: str = 'Symbol Price',
                   on_candles_indicator: list = [],
                   indicator_series: list = [],
                   indicator_names: list = [],
                   indicator_colors: list = [],
                   fill_control: dict or list = None,
                   indicators_filled_mode: dict or list = None,
                   axis_groups: dict or list = {},
                   plot_splitted_serie_couple: dict or list = {},
                   rows_pos: list = [],
                   plot_bgcolor=None,
                   actions_col: str = None,
                   priced_actions_col='Close',
                   labels: list = [],
                   markers: list = None,
                   marker_colors: list = None,
                   marker_legend_name: list = None):
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
    :param int width: Plot sizing
    :param int height: Plot sizing
    :param float candles_ta_height_ratio: A ratio between the big candles plot and (if any) the rest of indicator subplots below.
    :param bool plot_volume: Optional to plot volume.
    :param str title: A title string.
    :param str yaxis_title: A name string.
    :param on_candles_indicator: A list of pandas series with values to plot overlapping candles, not in a subplot. Example: SMA.
    :param list indicator_series: a list of pandas series with float values as indicators. Usually not overlap with candles indicators.
        But to plot in a subplot.
    :param list indicator_names: Names to show in the plot. Defaults to series name.
    :param list indicator_colors: Color can be forced to anyone from the plotly colors list:

         https://community.plotly.com/t/plotly-colours-list/11730
    :param dict or list fill_control: A dictionary with color to fill or False bool for each indicator. Is the color to the zero line for
        the indicator plot. If a list passed, it iterates to assign each item in the list with the same index item in the indicators list.
    :param dict indicators_filled_mode: A dict with filled areas for plotting.
    :param dict axis_groups: A dict with named groups for indicators, useful for plotting filled areas using tonexty fill mode.
    :param dict plot_splitted_serie_couple: A dict with splitted data for multiple colours when filling areas using tonexty.
    :param list rows_pos: 1 means over the candles. Other numbers mean subsequent subplots under the candles.
    :param plot_bgcolor: Set background color.
    :param actions_col: A column name of the column with string tags like buy, sell, etc. This is for plotting annotation marks
        overlapped over candles. It is *mandatory* for managing markers, annotations and legend names of annotations.
    :param priced_actions_col: The name of the column containing value of action to position over candles.

        Example:

            .. code-block:: python

                from binpan import binpan
                from handlers.strategies import random_strategy

                btcusdt = binpan.Symbol(symbol='btcusdt',
                                tick_interval='15m',
                                time_zone='Europe/Madrid',
                                end_time='2021-10-31 03:00:00')

                btcusdt.sma(21)


                df = random_strategy(data=btcusdt.df, buys_qty=10, sells_qty=12)

                df.actions.value_counts()

               sell     10
               buy      10
               Name: actions, dtype: int64

                binpan.handlers.plotting.candles_tagged(data=df,
                                                       plot_volume=False,
                                                       on_candles_indicator=[df.SMA_21],
                                                       candles_ta_height_ratio=0.8,
                                                       actions_col='actions',
                                                       labels=['buy', 'sell'])


            .. image:: images/plot_tagged_example_02.png
               :width: 1000

    :param list labels: Ordered like the annotations list of tags to show overlapped. Position defaults to close price. Example:

        .. code-block:: python

          labels = ['buy', 'sell']


    :param list markers: Plotly marker type. Usually, if referenced by number will be a not filled mark and using string name will be
        a color filled one. Check plotly info: https://plotly.com/python/marker-style/

        .. code-block::

            plotly_markers = [0, '0', 'circle', 100, '100', 'circle-open', 200, '200',
                    'circle-dot', 300, '300', 'circle-open-dot', 1, '1',
                    'square', 101, '101', 'square-open', 201, '201',
                    'square-dot', 301, '301', 'square-open-dot', 2, '2',
                    'diamond', 102, '102', 'diamond-open', 202, '202',
                    'diamond-dot', 302, '302', 'diamond-open-dot', 3, '3',
                    'cross', 103, '103', 'cross-open', 203, '203',
                    'cross-dot', 303, '303', 'cross-open-dot', 4, '4', 'x',
                    104, '104', 'x-open', 204, '204', 'x-dot', 304, '304',
                    'x-open-dot', 5, '5', 'triangle-up', 105, '105',
                    'triangle-up-open', 205, '205', 'triangle-up-dot', 305,
                    '305', 'triangle-up-open-dot', 6, '6', 'triangle-down',
                    106, '106', 'triangle-down-open', 206, '206',
                    'triangle-down-dot', 306, '306', 'triangle-down-open-dot',
                    7, '7', 'triangle-left', 107, '107', 'triangle-left-open',
                    207, '207', 'triangle-left-dot', 307, '307',
                    'triangle-left-open-dot', 8, '8', 'triangle-right', 108,
                    '108', 'triangle-right-open', 208, '208',
                    'triangle-right-dot', 308, '308',
                    'triangle-right-open-dot', 9, '9', 'triangle-ne', 109,
                    '109', 'triangle-ne-open', 209, '209', 'triangle-ne-dot',
                    309, '309', 'triangle-ne-open-dot', 10, '10',
                    'triangle-se', 110, '110', 'triangle-se-open', 210, '210',
                    'triangle-se-dot', 310, '310', 'triangle-se-open-dot', 11,
                    '11', 'triangle-sw', 111, '111', 'triangle-sw-open', 211,
                    '211', 'triangle-sw-dot', 311, '311',
                    'triangle-sw-open-dot', 12, '12', 'triangle-nw', 112,
                    '112', 'triangle-nw-open', 212, '212', 'triangle-nw-dot',
                    312, '312', 'triangle-nw-open-dot', 13, '13', 'pentagon',
                    113, '113', 'pentagon-open', 213, '213', 'pentagon-dot',
                    313, '313', 'pentagon-open-dot', 14, '14', 'hexagon', 114,
                    '114', 'hexagon-open', 214, '214', 'hexagon-dot', 314,
                    '314', 'hexagon-open-dot', 15, '15', 'hexagon2', 115,
                    '115', 'hexagon2-open', 215, '215', 'hexagon2-dot', 315,
                    '315', 'hexagon2-open-dot', 16, '16', 'octagon', 116,
                    '116', 'octagon-open', 216, '216', 'octagon-dot', 316,
                    '316', 'octagon-open-dot', 17, '17', 'star', 117, '117',
                    'star-open', 217, '217', 'star-dot', 317, '317',
                    'star-open-dot', 18, '18', 'hexagram', 118, '118',
                    'hexagram-open', 218, '218', 'hexagram-dot', 318, '318',
                    'hexagram-open-dot', 19, '19', 'star-triangle-up', 119,
                    '119', 'star-triangle-up-open', 219, '219',
                    'star-triangle-up-dot', 319, '319',
                    'star-triangle-up-open-dot', 20, '20',
                    'star-triangle-down', 120, '120',
                    'star-triangle-down-open', 220, '220',
                    'star-triangle-down-dot', 320, '320',
                    'star-triangle-down-open-dot', 21, '21', 'star-square',
                    121, '121', 'star-square-open', 221, '221',
                    'star-square-dot', 321, '321', 'star-square-open-dot', 22,
                    '22', 'star-diamond', 122, '122', 'star-diamond-open',
                    222, '222', 'star-diamond-dot', 322, '322',
                    'star-diamond-open-dot', 23, '23', 'diamond-tall', 123,
                    '123', 'diamond-tall-open', 223, '223',
                    'diamond-tall-dot', 323, '323', 'diamond-tall-open-dot',
                    24, '24', 'diamond-wide', 124, '124', 'diamond-wide-open',
                    224, '224', 'diamond-wide-dot', 324, '324',
                    'diamond-wide-open-dot', 25, '25', 'hourglass', 125,
                    '125', 'hourglass-open', 26, '26', 'bowtie', 126, '126',
                    'bowtie-open', 27, '27', 'circle-cross', 127, '127',
                    'circle-cross-open', 28, '28', 'circle-x', 128, '128',
                    'circle-x-open', 29, '29', 'square-cross', 129, '129',
                    'square-cross-open', 30, '30', 'square-x', 130, '130',
                    'square-x-open', 31, '31', 'diamond-cross', 131, '131',
                    'diamond-cross-open', 32, '32', 'diamond-x', 132, '132',
                    'diamond-x-open', 33, '33', 'cross-thin', 133, '133',
                    'cross-thin-open', 34, '34', 'x-thin', 134, '134',
                    'x-thin-open', 35, '35', 'asterisk', 135, '135',
                    'asterisk-open', 36, '36', 'hash', 136, '136',
                    'hash-open', 236, '236', 'hash-dot', 336, '336',
                    'hash-open-dot', 37, '37', 'y-up', 137, '137',
                    'y-up-open', 38, '38', 'y-down', 138, '138',
                    'y-down-open', 39, '39', 'y-left', 139, '139',
                    'y-left-open', 40, '40', 'y-right', 140, '140',
                    'y-right-open', 41, '41', 'line-ew', 141, '141',
                    'line-ew-open', 42, '42', 'line-ns', 142, '142',
                    'line-ns-open', 43, '43', 'line-ne', 143, '143',
                    'line-ne-open', 44, '44', 'line-nw', 144, '144',
                    'line-nw-open', 45, '45', 'arrow-up', 145, '145',
                    'arrow-up-open', 46, '46', 'arrow-down', 146, '146',
                    'arrow-down-open', 47, '47', 'arrow-left', 147, '147',
                    'arrow-left-open', 48, '48', 'arrow-right', 148, '148',
                    'arrow-right-open', 49, '49', 'arrow-bar-up', 149, '149',
                    'arrow-bar-up-open', 50, '50', 'arrow-bar-down', 150,
                    '150', 'arrow-bar-down-open', 51, '51', 'arrow-bar-left',
                    151, '151', 'arrow-bar-left-open', 52, '52',
                    'arrow-bar-right', 152, '152', 'arrow-bar-right-open']

    :param list marker_colors: Colors of the annotations.
    :param list marker_legend_name: A list with the names to print as tags over the annotations.

    """
    data_ = data.copy(deep=True)
    annotations_values = []

    if type(fill_control) == list:
        fill_control = {s.name: fill_control[i] for i, s in enumerate(indicator_series)}

    if type(indicators_filled_mode) == list:
        indicators_filled_mode = {s.name: indicators_filled_mode[i] for i, s in enumerate(indicator_series)}

    if actions_col:  # this trigger all annotation and markers thing

        # check type
        data_string_actions_col = data_[actions_col].astype('string')

        if not labels:
            labels = list(data_string_actions_col.value_counts().index)

        if not markers:
            # markers = ["arrow-bar-up", "arrow-bar-down"]
            markers = ["arrow-left" for _ in range(len(labels))]

        if not marker_colors:
            marker_colors = [choice(plotly_colors) for _ in range(len(labels))]

        if not marker_legend_name:
            try:
                marker_legend_name = [i[0].upper() + i[1:] for i in labels]
            except Exception as exc:
                raise BinPanException(exc.__class__, "Actions are not subscriptable.", "Check actions column is a strings column.")

        actions = list(data_string_actions_col.value_counts().index)
        for action in actions:
            annotations_values.append(data_[data_[actions_col] == action][priced_actions_col])

        # verify annotations, colors, labels and names
        try:
            assert len(labels) == len(markers)
            assert len(labels) == len(marker_colors)
            assert len(labels) == len(marker_legend_name)
            # assert len(annotations_values) == len(data[actions_col].dropna())

        except Exception as exc:
            raise BinPanException(exc.__class__,
                                  "Plotting labels, annotation colors or names not consistent with markers list length",
                                  "Function candles_tagged")

    # indicator allocating rows
    rows_pos_final = []

    if on_candles_indicator:
        rows_pos_final = [1 for _ in range(len(on_candles_indicator))]

    if indicator_series and not rows_pos:
        rows_pos_final += [i + 2 for i in range(len(indicator_series))]
    else:
        rows_pos_final += rows_pos + [i + 2 for i in range(len(indicator_series) - len(rows_pos))]

    indicator_series = on_candles_indicator + indicator_series

    # indicator names for legend
    if not indicator_names:
        try:
            indicator_names = [i.name for i in indicator_series]
        except Exception:
            indicator_names = []
            for i, ind in enumerate(indicator_series):
                try:
                    indicator_names.append(ind.name)
                except:
                    indicator_names.append(f'Indicator_{i}')

    candles_ta(data_,
               width=width,
               height=height,
               range_slider=False,
               candles_ta_height_ratio=candles_ta_height_ratio,
               plot_volume=plot_volume,
               title=title,
               yaxis_title=yaxis_title,
               annotation_values=annotations_values,
               markers=markers,
               rows_pos=rows_pos_final,
               labels=labels,
               annotation_colors=marker_colors,
               annotation_legend_names=marker_legend_name,
               indicators_series=indicator_series,
               indicator_names=indicator_names,
               indicators_colors=indicator_colors,
               indicators_color_filled=fill_control,
               indicators_filled_mode=indicators_filled_mode,
               axis_groups=axis_groups,
               plot_splitted_serie_couple=plot_splitted_serie_couple,
               plot_bgcolor=plot_bgcolor)


################
# trades plots #
################

def plot_trade_size(data: pd.DataFrame,
                    max_size: int = 60,
                    height: int = 1000,
                    logarithmic: bool = False,
                    title: str = None,
                    **kwargs_update_layout):
    """
    Plots scatter plot from trades quantity and trades sizes. Marks are size scaled to the max size. Marks are semi transparent and colored
    using Maker buyer or Taker buyer discrete colors. Usually red and blue.

    Can let you see where are the big sized trades done and the taker or maker buyer side.

    :param pd.DataFrame data: A BinPans trades dataframe.
    :param int max_size: Size of the marks for the biggest quantity sized trades.
    :param int height: Plot sizing.
    :param bool logarithmic: Y axis in a logarithmic scale.
    :param str title: Title string.
    :param kwargs_update_layout: Update layout plotly options.

    Example:
        .. code-block:: python

           from binpan import binpan

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
                     size_max=max_size, log_y=logarithmic)
    if not title:
        title = f"Trades size {data.index.name}"
    fig.update_layout(
        title=title,
        xaxis_title_text=f'{data.index.name}',
        yaxis_title_text=f'Price',
        height=height,
        **kwargs_update_layout)
    fig.show()


def normalize(max_value, min_value, data: list):
    return [(i / sum(data)) * max_value + min_value for i in data]


###################
# analyzing plots #
###################

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

           from binpan import binpan

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

       from binpan import binpan

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

       from binpan import binpan

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

        from binpan import binpan

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


def dist_plot(df: pd.DataFrame,
              x_col: str = 'Price',
              color: str = 'Side',
              bins: int = 300,
              histnorm: str = 'density',
              height: int = 800,
              title: str = "Distribution",
              **update_layout_kwargs):
    """
    Plot a distribution plot for a dataframe column. Plots line for kernel distribution.

    :param pd.DataFrame df: A BinPan Dataframe like orderbook, candles, or any other.
    :param str x_col: Column name for x-axis data.
    :param str color: Column name with tags or any values for using as color scale.
    :param int bins: Columns in histogram.
    :param str histnorm: One of 'percent', 'probability', 'density', or 'probability density' from plotly express documentation.
        https://plotly.github.io/plotly.py-docs/generated/plotly.express.histogram.html
    :param int height: Plot sizing.
    :param str title: A title string

    Example from binpan Symbol plot_orderbook_density method.

    .. image:: images/orderbook_density.png
       :width: 1000
       :alt: Plot example

    """
    filtered_df = df.copy()

    fig = ff.create_distplot(hist_data=[filtered_df["Price"].tolist()],
                             group_labels=["Price"],
                             show_hist=False,
                             ).add_traces(
        px.histogram(filtered_df, x=x_col, nbins=bins, color=color, histnorm=histnorm)
        .update_traces(yaxis="y3", name=x_col)
        .data)

    fig.update_layout(height=height, title=title, yaxis3={"overlaying": "y", "side": "right"}, showlegend=True, **update_layout_kwargs)
    fig.show()
