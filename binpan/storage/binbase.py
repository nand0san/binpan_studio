"""
Cliente de lectura para binbase (TimescaleDB).

binbase almacena datos de mercado de Binance en tiempo real en un esquema
compartido con columna ``symbol``:

- ``trades`` — hypertable de trades atómicos
- ``agg_trades`` — hypertable de trades agregados
- ``orderbook_snapshots`` — hypertable de snapshots de profundidad (cada 5s)
- ``klines_*`` — continuous aggregates derivados de trades (1m, 5m, 1h, 1d …)

Todas las tablas/vistas usan columna ``symbol`` en vez de una tabla por par.
BinPan solo lee; nunca escribe en binbase.
"""

import os

import pandas as pd
import psycopg2

from ..core.exceptions import BinPanException
from ..core.logs import LogManager
from ..core.time_helper import tick_seconds
from ..core.standards import (
    kline_open_time_col, kline_open_col, kline_high_col, kline_low_col,
    kline_close_col, kline_volume_col, kline_quote_volume_col,
    kline_trades_col, kline_taker_buy_base_volume_col,
    kline_taker_buy_quote_volume_col, kline_open_timestamp_col,
    kline_close_time_col, kline_close_timestamp_col,
    trade_trade_id_col, trade_price_col, trade_quantity_col,
    trade_quote_quantity_col, trade_timestamp_col, trade_date_col,
    trade_buyer_was_maker_col,
    agg_trade_id_col, agg_trade_price_col, agg_quantity_col,
    agg_first_trade_id_col, agg_last_trade_id_col,
    agg_date_col, agg_time_col, agg_buyer_was_maker_col,
)
from ..api.market import convert_to_numeric

sql_logger = LogManager(filename='./logs/sql.log', name='binbase', info_level='INFO')

# ============================================================================
# Defaults (solo red local 192.168.100.x)
# ============================================================================

BINBASE_HOST = "192.168.100.221"
BINBASE_PORT = 5432
BINBASE_USER = "binbase"
BINBASE_DATABASE = "binbase"

# Intervalos disponibles como vistas klines_* en binbase
BINBASE_INTERVALS = frozenset({
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w",
})

# ============================================================================
# Mapeos de columnas: binbase -> BinPan
# ============================================================================

KLINE_COLUMN_MAP = {
    "bucket":                kline_open_time_col,
    "open":                  kline_open_col,
    "high":                  kline_high_col,
    "low":                   kline_low_col,
    "close":                 kline_close_col,
    "volume":                kline_volume_col,
    "quote_volume":          kline_quote_volume_col,
    "trades":                kline_trades_col,
    "taker_buy_volume":      kline_taker_buy_base_volume_col,
    "taker_buy_quote_volume": kline_taker_buy_quote_volume_col,
}

TRADE_COLUMN_MAP = {
    "time":         trade_timestamp_col,
    "trade_id":     trade_trade_id_col,
    "price":        trade_price_col,
    "quantity":     trade_quantity_col,
    "quote_qty":    trade_quote_quantity_col,
    "buyer_maker":  trade_buyer_was_maker_col,
}

AGG_TRADE_COLUMN_MAP = {
    "time":           agg_time_col,
    "agg_trade_id":   agg_trade_id_col,
    "price":          agg_trade_price_col,
    "quantity":       agg_quantity_col,
    "first_trade_id": agg_first_trade_id_col,
    "last_trade_id":  agg_last_trade_id_col,
    "buyer_maker":    agg_buyer_was_maker_col,
}

# ============================================================================
# Conexión
# ============================================================================


def _resolve_password(password: str | None) -> str:
    """Resuelve password: argumento > secret.py > variable de entorno."""
    if password:
        return password
    try:
        from secret import binbase_password
        return binbase_password
    except (ImportError, AttributeError):
        pass
    env = os.environ.get("BINBASE_PASSWORD")
    if env:
        return env
    raise BinPanException(
        "binbase: password requerido. Opciones:\n"
        "  1) Pasar password= al conectar\n"
        "  2) Definir binbase_password en secret.py\n"
        "  3) Exportar BINBASE_PASSWORD como variable de entorno"
    )


def connect(host: str = BINBASE_HOST,
            port: int = BINBASE_PORT,
            user: str = BINBASE_USER,
            password: str = None,
            database: str = BINBASE_DATABASE,
            timeout: int = 10) -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Connect to binbase TimescaleDB (read-only).

    :param str host: Host address. Default ``192.168.100.221``.
    :param int port: Port. Default ``5432``.
    :param str user: User. Default ``binbase``.
    :param str password: Password. Falls back to ``secret.binbase_password`` or ``BINBASE_PASSWORD`` env var.
    :param str database: Database name. Default ``binbase``.
    :param int timeout: Connection timeout in seconds.
    :return: ``(connection, cursor)`` tuple.
    """
    password = _resolve_password(password)
    try:
        connection = psycopg2.connect(
            host=host, port=port, user=user,
            password=password, database=database,
            connect_timeout=timeout,
        )
        connection.autocommit = True
        cursor = connection.cursor()
        sql_logger.debug(f"Conectado a binbase en {host}:{port}/{database}")
        return connection, cursor
    except Exception as e:
        raise BinPanException(f"binbase: error de conexión a {host}:{port}/{database} — {e}")


def close(connection) -> None:
    """Close a binbase connection."""
    if connection and not connection.closed:
        connection.close()


# ============================================================================
# Consultas
# ============================================================================


def get_symbols(cursor) -> list[str]:
    """
    Return list of symbols available in binbase.

    :param cursor: psycopg2 cursor.
    :return: Sorted list of symbol strings.
    """
    cursor.execute("SELECT DISTINCT symbol FROM trades ORDER BY symbol;")
    return [row[0] for row in cursor.fetchall()]


def get_klines(cursor,
               symbol: str,
               tick_interval: str,
               start_time: int,
               end_time: int,
               time_zone: str = "UTC") -> pd.DataFrame:
    """
    Fetch klines from a binbase continuous aggregate view.

    Returns a DataFrame compatible with BinPan's kline format, including
    computed ``Open timestamp``, ``Close timestamp`` and ``Close time`` columns.

    :param cursor: psycopg2 cursor.
    :param str symbol: Trading pair (e.g. ``'BTCUSDC'``).
    :param str tick_interval: Candle interval (e.g. ``'1h'``).
    :param int start_time: Start timestamp in milliseconds.
    :param int end_time: End timestamp in milliseconds.
    :param str time_zone: IANA timezone for the index.
    :return: DataFrame with BinPan-standard kline columns.
    """
    if tick_interval not in BINBASE_INTERVALS:
        raise BinPanException(
            f"binbase: intervalo '{tick_interval}' no disponible. "
            f"Válidos: {sorted(BINBASE_INTERVALS)}"
        )

    view = f"klines_{tick_interval}"
    bb_cols = list(KLINE_COLUMN_MAP.keys())
    col_list = ", ".join(bb_cols)

    query = (
        f"SELECT {col_list} FROM {view} "
        "WHERE symbol = %s "
        "  AND bucket >= to_timestamp(%s / 1000.0) "
        "  AND bucket < to_timestamp(%s / 1000.0) "
        "ORDER BY bucket"
    )
    sql_logger.info(f"binbase klines: {symbol} {tick_interval} [{start_time} → {end_time}]")
    cursor.execute(query, (symbol.upper(), start_time, end_time))
    data = cursor.fetchall()

    if not data:
        binpan_cols = [KLINE_COLUMN_MAP[c] for c in bb_cols] + [kline_open_timestamp_col,
                                                                  kline_close_timestamp_col,
                                                                  kline_close_time_col]
        return pd.DataFrame(columns=binpan_cols)

    df = pd.DataFrame(data, columns=bb_cols)
    df = df.rename(columns=KLINE_COLUMN_MAP)

    # bucket viene como datetime tz-aware desde psycopg2
    df[kline_open_time_col] = pd.to_datetime(df[kline_open_time_col], utc=True)
    df[kline_open_time_col] = df[kline_open_time_col].dt.tz_convert(time_zone)
    df[kline_open_timestamp_col] = (df[kline_open_time_col].astype('int64') // 10 ** 6).astype('int64')

    # close timestamp sintético: open + intervalo - 1ms (convención Binance)
    interval_ms = tick_seconds[tick_interval] * 1000
    df[kline_close_timestamp_col] = df[kline_open_timestamp_col] + interval_ms - 1
    df[kline_close_time_col] = pd.to_datetime(df[kline_close_timestamp_col], unit='ms', utc=True)
    df[kline_close_time_col] = df[kline_close_time_col].dt.tz_convert(time_zone)

    df = df.set_index(kline_open_time_col, drop=False)
    df = df.sort_index()
    df.index.name = f"{symbol.upper()} {tick_interval} {time_zone}"

    df = convert_to_numeric(data=df)
    return df


def get_trades(cursor,
               symbol: str,
               start_time: int,
               end_time: int,
               time_zone: str = "UTC") -> pd.DataFrame:
    """
    Fetch atomic trades from binbase.

    :param cursor: psycopg2 cursor.
    :param str symbol: Trading pair (e.g. ``'BTCUSDC'``).
    :param int start_time: Start timestamp in milliseconds.
    :param int end_time: End timestamp in milliseconds.
    :param str time_zone: IANA timezone for the index.
    :return: DataFrame with BinPan-standard trade columns.
    """
    bb_cols = list(TRADE_COLUMN_MAP.keys())
    col_list = ", ".join(bb_cols)

    query = (
        f"SELECT {col_list} FROM trades "
        "WHERE symbol = %s "
        "  AND time >= to_timestamp(%s / 1000.0) "
        "  AND time < to_timestamp(%s / 1000.0) "
        "ORDER BY time"
    )
    sql_logger.info(f"binbase trades: {symbol} [{start_time} → {end_time}]")
    cursor.execute(query, (symbol.upper(), start_time, end_time))
    data = cursor.fetchall()

    if not data:
        binpan_cols = [TRADE_COLUMN_MAP[c] for c in bb_cols] + [trade_date_col]
        return pd.DataFrame(columns=binpan_cols)

    df = pd.DataFrame(data, columns=bb_cols)
    df = df.rename(columns=TRADE_COLUMN_MAP)

    # time viene como datetime tz-aware desde psycopg2
    dt = pd.to_datetime(df[trade_timestamp_col], utc=True)
    df[trade_date_col] = dt.dt.tz_convert(time_zone)
    df[trade_timestamp_col] = (dt.astype('int64') // 10 ** 6).astype('int64')

    df = df.sort_values([trade_date_col, trade_trade_id_col])
    df = df.set_index(trade_date_col, drop=False)
    df.index.name = f"{symbol.upper()} {time_zone}"

    df = convert_to_numeric(data=df)
    return df


def get_agg_trades(cursor,
                   symbol: str,
                   start_time: int,
                   end_time: int,
                   time_zone: str = "UTC") -> pd.DataFrame:
    """
    Fetch aggregated trades from binbase.

    :param cursor: psycopg2 cursor.
    :param str symbol: Trading pair (e.g. ``'BTCUSDC'``).
    :param int start_time: Start timestamp in milliseconds.
    :param int end_time: End timestamp in milliseconds.
    :param str time_zone: IANA timezone for the index.
    :return: DataFrame with BinPan-standard agg trade columns.
    """
    bb_cols = list(AGG_TRADE_COLUMN_MAP.keys())
    col_list = ", ".join(bb_cols)

    query = (
        f"SELECT {col_list} FROM agg_trades "
        "WHERE symbol = %s "
        "  AND time >= to_timestamp(%s / 1000.0) "
        "  AND time < to_timestamp(%s / 1000.0) "
        "ORDER BY time"
    )
    sql_logger.info(f"binbase agg_trades: {symbol} [{start_time} → {end_time}]")
    cursor.execute(query, (symbol.upper(), start_time, end_time))
    data = cursor.fetchall()

    if not data:
        binpan_cols = [AGG_TRADE_COLUMN_MAP[c] for c in bb_cols] + [agg_date_col]
        return pd.DataFrame(columns=binpan_cols)

    df = pd.DataFrame(data, columns=bb_cols)
    df = df.rename(columns=AGG_TRADE_COLUMN_MAP)

    # time viene como datetime tz-aware desde psycopg2
    dt = pd.to_datetime(df[agg_time_col], utc=True)
    df[agg_date_col] = dt.dt.tz_convert(time_zone)
    df[agg_time_col] = (dt.astype('int64') // 10 ** 6).astype('int64')

    df = df.sort_values([agg_date_col, agg_trade_id_col])
    df = df.set_index(agg_date_col, drop=False)
    df.index.name = f"{symbol.upper()} {time_zone}"

    df = convert_to_numeric(data=df)
    return df


def get_orderbook(cursor,
                  symbol: str,
                  start_time: int = None,
                  end_time: int = None,
                  last: bool = False) -> pd.DataFrame:
    """
    Fetch orderbook snapshots from binbase.

    :param cursor: psycopg2 cursor.
    :param str symbol: Trading pair (e.g. ``'BTCUSDC'``).
    :param int start_time: Start timestamp in milliseconds (ignored if ``last=True``).
    :param int end_time: End timestamp in milliseconds (ignored if ``last=True``).
    :param bool last: If True, return only the most recent snapshot.
    :return: DataFrame with ``time``, ``last_update_id``, ``bids``, ``asks`` columns.
    """
    if last:
        query = (
            "SELECT time, last_update_id, bids, asks "
            "FROM orderbook_snapshots "
            "WHERE symbol = %s "
            "ORDER BY time DESC LIMIT 1"
        )
        cursor.execute(query, (symbol.upper(),))
    else:
        if start_time is None or end_time is None:
            raise BinPanException("binbase get_orderbook: start_time y end_time requeridos si last=False")
        query = (
            "SELECT time, last_update_id, bids, asks "
            "FROM orderbook_snapshots "
            "WHERE symbol = %s "
            "  AND time >= to_timestamp(%s / 1000.0) "
            "  AND time < to_timestamp(%s / 1000.0) "
            "ORDER BY time"
        )
        cursor.execute(query, (symbol.upper(), start_time, end_time))

    data = cursor.fetchall()
    if not data:
        return pd.DataFrame(columns=["time", "last_update_id", "bids", "asks"])

    df = pd.DataFrame(data, columns=["time", "last_update_id", "bids", "asks"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df
