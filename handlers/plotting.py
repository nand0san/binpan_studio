import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from random import choice
from .logs import Logs

plot_logger = Logs(filename='./logs/plotting.log', name='plotting', info_level='DEBUG')

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


def set_ta_line(df, serie, color='blue', name='Indicator', width=0.5):
    return go.Scatter(x=df.index, y=serie, line=dict(color=color, width=width), name=name, mode='lines')


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


def add_traces(fig, list_of_plots, rows, cols):
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
               labels: list = None):
    """Se pasan los datos junto con una lista de series de mismo índice con los indicadores.
    El campo row_pos indica en que fila se van a plotear los indicadores. Hay que tener en cuenta que si se ha marcado
    como True el ploteado del volume, este se introduce en la fila 2 y se van a desplazar el resto de indicadores
    automáticamente.
    Ejemplo: rows_pos = [2, 3, 4] para tres indicadores, usaría fila 1 para el tema de velas y si se añade volumen las
    filas para los indicadores pasarían a ser 3,4,5.
    """
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
    for i, indicator in enumerate(indicators_series):
        tas.append(set_ta_line(df=df_plot, serie=indicator, color=indicators_colors[i], name=indicator_names[i],
                               width=1))
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
                   rows_pos: list = [],
                   labels: list = [],
                   default_price_for_actions='Close'):
    """Shortcut para plotear rápido la estrategia buy/sell"""
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
               labels=labels)


def plot_trade_size(data: pd.DataFrame, max_size=60, height=1000, logarithmic=False, title=f"Trade Size"):
    data['Buyer was maker'].replace({True: 'Maker buyer', False: 'Taker buyer'}, inplace=True)
    fig = px.scatter(x=data.index, y=data['Price'], color=data['Buyer was maker'], size=data['Quantity'],
                     title=title,
                     height=height, size_max=max_size, log_y=logarithmic)
    fig.show()


def normalize(max_value, min_value, data: list):
    return [(i / sum(data)) * max_value + min_value for i in data]


def plot_pie(serie: pd.Series, categories: int = 15, title=f"Size trade categories", logarithmic=False):
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

    orders = {serie.name: spread}

    pie = serie.groupby(pd.cut(serie, spread)).count()
    names = [str(i) for i in pie.index]

    fig = px.pie(pie,
                 # values='Quantity',
                 values=serie.name,
                 names=names,
                 color_discrete_sequence=px.colors.sequential.RdBu,
                 title=title,
                 hover_name=serie.name,
                 category_orders=orders)
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


def plot_hists_vs(x0,
                  x1,
                  x0_name: str = None,
                  x1_name: str = None,
                  bins=50,
                  hist_funct='sum',
                  height=900,
                  title: str = None,
                  **kwargs_update_layout):

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
