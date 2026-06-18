"""
Trades value object.

Wraps a trades :class:`pandas.DataFrame` together with its metadata (trade type,
data origin, the column map and raw API response it was built from). A single
``Symbol`` can hold two independent instances at once, e.g. atomic trades from
one origin and aggregated trades from another.

It proxies attribute and item access to the underlying DataFrame, so it can be
used like one in notebooks::

    s.agg_trades['Quantity']     # -> pandas Series  (delegated to .df)
    s.agg_trades.empty           # -> bool
    s.agg_trades.head()          # -> DataFrame      (delegated to .df)
    s.agg_trades.df              # -> the raw pandas DataFrame
    s.agg_trades.trade_type      # -> 'agg'

Heavy domain logic (indicators, plotting) is NOT placed here to keep this module
dependency-free (pandas only) and avoid import cycles with ``analysis``/``plotting``.
Only the small, repeated bits live here: the empty check, the time-range filter
and the metadata.
"""

import pandas as pd

_AGG_TYPES = ("agg", "aggtrade", "aggTrade")


class Trades:
    """
    A trades DataFrame plus its metadata.

    :param pd.DataFrame df: The parsed trades DataFrame. If None, an empty frame
        is created (using ``columns`` for its column names when provided).
    :param str trade_type: ``'agg'`` (aggregated) or ``'atomic'``/``'trade'``.
    :param str origin: Where the data came from: ``'binance_api'``, ``'csv'``,
        ``'binbase'`` or ``'postgresql'``.
    :param dict columns: Column map (raw API field -> BinPan column name).
    :param str symbol: The symbol, e.g. ``'BTCUSDT'``.
    :param str time_zone: IANA time zone of the index, e.g. ``'Europe/Madrid'``.
    :param list raw: The raw API response (list of dicts) it was parsed from.
    """

    def __init__(self, df: pd.DataFrame = None, *, trade_type: str = "agg", origin: str = None,
                 columns: dict = None, symbol: str = None, time_zone: str = None, raw: list = None):
        if df is None:
            df = pd.DataFrame(columns=list(columns.values()) if columns else None)
        self.df = df
        self.trade_type = trade_type
        self.origin = origin
        self.columns = columns
        self.symbol = symbol
        self.time_zone = time_zone
        self.raw = raw if raw is not None else []

    # ── metadata helpers ─────────────────────────────────────

    @property
    def is_agg(self) -> bool:
        """True if these are aggregated trades."""
        return self.trade_type in _AGG_TYPES

    @property
    def label(self) -> str:
        """Human label: ``'aggregated'`` or ``'atomic'``."""
        return "aggregated" if self.is_agg else "atomic"

    @property
    def empty_msg(self) -> str:
        """Message suggesting how to fetch the data when empty."""
        method = "get_agg_trades()" if self.is_agg else "get_atomic_trades()"
        return (f"Empty {self.label} trades, please request using: {method} method. "
                f"Example: my_symbol.{method}")

    def filtered(self, start_time: int = None, end_time: int = None, time_col: str = "Timestamp") -> pd.DataFrame:
        """
        Return a copy of the DataFrame filtered by the timestamp column.

        :param int start_time: Inclusive lower bound (ms). Ignored if falsy.
        :param int end_time: Inclusive upper bound (ms). Ignored if falsy.
        :param str time_col: Timestamp column name. Default ``'Timestamp'``.
        :return pd.DataFrame: Filtered (deep) copy.
        """
        df = self.df.copy(deep=True)
        if start_time:
            df = df[df[time_col] >= start_time]
        if end_time:
            df = df[df[time_col] <= end_time]
        return df

    def with_df(self, df: pd.DataFrame) -> "Trades":
        """Return a new Trades carrying the same metadata but a different DataFrame."""
        return Trades(df, trade_type=self.trade_type, origin=self.origin, columns=self.columns,
                      symbol=self.symbol, time_zone=self.time_zone, raw=self.raw)

    # ── DataFrame proxy ──────────────────────────────────────

    @property
    def empty(self) -> bool:
        """True if the underlying DataFrame is empty."""
        return self.df.empty

    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails -> delegate to the DataFrame.
        # Magic methods are intentionally not delegated.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            df = object.__getattribute__(self, "df")
        except AttributeError:
            raise AttributeError(name)
        return getattr(df, name)

    def __getitem__(self, key):
        return self.df[key]

    def __setitem__(self, key, value):
        self.df[key] = value

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self):
        return iter(self.df)

    def __contains__(self, item) -> bool:
        return item in self.df

    def __repr__(self) -> str:
        return (f"Trades(symbol={self.symbol}, type={self.trade_type}, "
                f"origin={self.origin}, rows={len(self.df)})")

    def _repr_html_(self):
        return self.df._repr_html_()
