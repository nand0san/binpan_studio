Plotting Module
===============

This module can manage plots.

Plots made with Plotly library: https://plotly.com/

To import this module:

.. code-block::

   from handlers import plotting


.. automodule:: handlers.plotting

Colors
------

Colors can be picked with the name of the color or the index number from this list:

.. code-block::

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

Candles Plots
-------------

.. autofunction:: candles_ta

.. autofunction:: candles_tagged

Trades Plots
------------

.. autofunction:: plot_trade_size

Analysis Plots
--------------

.. autofunction:: plot_pie

.. autofunction:: plot_scatter

.. autofunction:: plot_hists_vs

.. autofunction:: orderbook_depth

.. autofunction:: dist_plot


