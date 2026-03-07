"""
Formatos de datos de la API REST de Binance y mapeos internos de BinPan.

Fuente: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md
Verificado contra la API real el 2026-03-07.

Endpoints cubiertos:
    - GET /api/v3/klines (y /api/v3/uiKlines, mismo formato)
    - GET /api/v3/aggTrades
    - GET /api/v3/trades
    - GET /api/v3/historicalTrades (mismo formato que /api/v3/trades)
    - GET /api/v3/depth
    - GET /api/v3/ticker/bookTicker
    - GET /api/v3/ticker/price

Solo se usa el market "spot". No hay soporte de futuros (fapi/dapi).
"""

# ============================================================================
# BINANCE API - FORMATOS DE RESPUESTA
# ============================================================================
#
# Klines (GET /api/v3/klines)
# ---------------------------
# Respuesta: lista de listas, cada una con 12 elementos:
#
#   [0]  int  Open time (ms epoch)
#   [1]  str  Open price
#   [2]  str  High price
#   [3]  str  Low price
#   [4]  str  Close price
#   [5]  str  Volume (base asset)
#   [6]  int  Close time (ms epoch)
#   [7]  str  Quote asset volume
#   [8]  int  Number of trades
#   [9]  str  Taker buy base asset volume
#   [10] str  Taker buy quote asset volume
#   [11] str  Unused field ("0")
#
# AggTrades (GET /api/v3/aggTrades)
# ---------------------------------
# Respuesta: lista de dicts con 8 campos:
#
#   "a"  int   Aggregate tradeId
#   "p"  str   Price
#   "q"  str   Quantity
#   "f"  int   First tradeId
#   "l"  int   Last tradeId
#   "T"  int   Timestamp (ms epoch)
#   "m"  bool  Was the buyer the maker?
#   "M"  bool  Was the trade the best price match?
#
# Atomic trades (GET /api/v3/trades y /api/v3/historicalTrades)
# -------------------------------------------------------------
# Respuesta: lista de dicts con 7 campos:
#
#   "id"           int   Trade id
#   "price"        str   Price
#   "qty"          str   Quantity
#   "quoteQty"     str   Quote quantity
#   "time"         int   Timestamp (ms epoch)
#   "isBuyerMaker" bool  Was the buyer the maker?
#   "isBestMatch"  bool  Was the trade the best price match?
#
# Order Book (GET /api/v3/depth)
# ------------------------------
# Respuesta: dict con 3 campos:
#
#   "lastUpdateId"  int           Last update id
#   "bids"          list[list]    [[price_str, qty_str], ...]
#   "asks"          list[list]    [[price_str, qty_str], ...]
#
# Book Ticker (GET /api/v3/ticker/bookTicker)
# -------------------------------------------
# Respuesta: dict (si symbol) o lista de dicts con 5 campos:
#
#   "symbol"    str  Trading pair
#   "bidPrice"  str  Best bid price
#   "bidQty"    str  Best bid quantity
#   "askPrice"  str  Best ask price
#   "askQty"    str  Best ask quantity
#
# Price Ticker (GET /api/v3/ticker/price)
# ----------------------------------------
# Respuesta: dict (si symbol) o lista de dicts con 2 campos:
#
#   "symbol"  str  Trading pair
#   "price"   str  Current price
#

# ============================================================================
# TIMESCALE / STREAM
# ============================================================================

stream_uniqueness_id_in_timescale = {
    'aggTrade': 'trade_id',
    'aggtrade': 'trade_id',
    'trade': 'trade_id',
    'kline': 'time',
    'statistics': 'time',
}

# ============================================================================
# NOMBRES DE COLUMNA EN BINPAN (presentacion interna)
# ============================================================================

# -- Klines --

kline_open_time_col = 'Open time'
kline_open_col = 'Open'
kline_high_col = 'High'
kline_low_col = 'Low'
kline_close_col = 'Close'
kline_volume_col = 'Volume'
kline_close_time_col = 'Close time'
kline_quote_volume_col = 'Quote volume'
kline_trades_col = 'Trades'
kline_taker_buy_base_volume_col = 'Taker buy base volume'
kline_taker_buy_quote_volume_col = 'Taker buy quote volume'
kline_ignore_col = 'Ignore'
kline_open_timestamp_col = 'Open timestamp'
kline_close_timestamp_col = 'Close timestamp'
kline_first_trade_id_col = 'First TradeId'
kline_last_trade_id_col = 'Last TradeId'

# -- Trades (atomicos) --

trade_trade_id_col = 'Trade Id'
trade_price_col = 'Price'
trade_quantity_col = 'Quantity'
trade_quote_quantity_col = 'Quote quantity'
trade_date_col = 'Date'
trade_buyer_order_id_col = 'Buyer Order Id'
trade_seller_order_id_col = 'Seller Order Id'
trade_timestamp_col = 'Timestamp'
trade_first_trade_id = 'First tradeId'
trade_last_trade_id = 'Last tradeId'
trade_buyer_was_maker_col = 'Buyer was maker'
trade_best_price_match_col = 'Best price match'
trade_agg_trade_id_col = 'Aggregate tradeId'

# -- AggTrades (alias para reutilizar nombres compartidos) --

agg_trade_id_col = 'Aggregate tradeId'
agg_trade_price_col = trade_price_col
agg_quantity_col = trade_quantity_col
agg_first_trade_id_col = 'First tradeId'
agg_last_trade_id_col = 'Last tradeId'
agg_date_col = trade_date_col
agg_time_col = trade_timestamp_col
agg_buyer_was_maker_col = trade_buyer_was_maker_col
agg_best_price_match_col = trade_best_price_match_col

# ============================================================================
# MAPEOS API JSON -> COLUMNAS BINPAN
# ============================================================================

# Klines: claves WebSocket/Redis (json single-letter keys) -> columnas BinPan
# Se usa cuando los datos vienen como dicts (WebSocket/Redis), no como listas (REST).
klines_api_map_columns = {
    "t": kline_open_time_col,
    "o": kline_open_col,
    "h": kline_high_col,
    "l": kline_low_col,
    "c": kline_close_col,
    "v": kline_volume_col,
    "T": kline_close_time_col,
    "q": kline_quote_volume_col,
    "n": kline_trades_col,
    "V": kline_taker_buy_base_volume_col,
    "Q": kline_taker_buy_quote_volume_col,
    "B": kline_ignore_col,
}

# AggTrades: claves REST/WebSocket -> columnas BinPan
# Campos: a, p, q, f, l, T, m, M (8 campos)
agg_trades_api_map_columns = {
    'a': trade_agg_trade_id_col,
    'p': trade_price_col,
    'q': trade_quantity_col,
    'f': trade_first_trade_id,
    'l': trade_last_trade_id,
    'T': trade_timestamp_col,
    'm': trade_buyer_was_maker_col,
    'M': trade_best_price_match_col,
}

# Atomic trades: claves REST -> columnas BinPan
# Campos: id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch (7 campos)
atomic_trades_api_map_columns = {
    'id': trade_trade_id_col,
    'price': trade_price_col,
    'qty': trade_quantity_col,
    'quoteQty': trade_quote_quantity_col,
    'time': trade_timestamp_col,
    'isBuyerMaker': trade_buyer_was_maker_col,
    'isBestMatch': trade_best_price_match_col,
}

# Atomic trades: claves WebSocket/Redis -> columnas BinPan
# WebSocket incluye buyerOrderId y sellerOrderId pero NO quoteQty
atomic_trades_ws_map_columns = {
    't': trade_trade_id_col,
    'p': trade_price_col,
    'q': trade_quantity_col,
    'b': trade_buyer_order_id_col,
    'a': trade_seller_order_id_col,
    'T': trade_timestamp_col,
    'm': trade_buyer_was_maker_col,
    'M': trade_best_price_match_col,
}

# ============================================================================
# LISTAS DE COLUMNAS ORDENADAS (para DataFrames)
# ============================================================================

# Columnas del DataFrame de klines (12 de la API + 2 generadas por BinPan)
binance_api_candles_cols = [
    kline_open_time_col,            # [0] API
    kline_open_col,                 # [1] API
    kline_high_col,                 # [2] API
    kline_low_col,                  # [3] API
    kline_close_col,                # [4] API
    kline_volume_col,               # [5] API
    kline_close_time_col,           # [6] API
    kline_quote_volume_col,         # [7] API
    kline_trades_col,               # [8] API
    kline_taker_buy_base_volume_col,  # [9] API
    kline_taker_buy_quote_volume_col, # [10] API
    kline_ignore_col,               # [11] API
    kline_open_timestamp_col,       # generado por BinPan (copia de Open time en ms)
    kline_close_timestamp_col,      # generado por BinPan (copia de Close time en ms)
]

# Columnas del DataFrame de aggTrades
# Orden: ID, Price, Qty, First/Last tradeId, Date (generado), Timestamp, Maker, BestPrice
agg_trades_columns_from_binance = [
    trade_agg_trade_id_col,
    trade_price_col,
    trade_quantity_col,
    trade_first_trade_id,
    trade_last_trade_id,
    trade_date_col,                 # generado por BinPan (string legible de Timestamp)
    trade_timestamp_col,
    trade_buyer_was_maker_col,
    trade_best_price_match_col,
]

# AggTrades desde Redis (mismo orden, mismos campos)
agg_trades_columns_from_redis = agg_trades_columns_from_binance

# Columnas del DataFrame de atomic trades (desde REST API)
atomic_trades_columns_from_binance = [
    trade_trade_id_col,
    trade_price_col,
    trade_quantity_col,
    trade_quote_quantity_col,       # presente en REST, ausente en WebSocket
    trade_date_col,                 # generado por BinPan
    trade_timestamp_col,
    trade_buyer_was_maker_col,
    trade_best_price_match_col,
]

# Columnas del DataFrame de atomic trades (desde WebSocket/Redis)
# WebSocket incluye buyer/seller order IDs pero NO quoteQty
atomic_trades_columns_from_redis = [
    trade_trade_id_col,
    trade_price_col,
    trade_quantity_col,
    trade_buyer_order_id_col,       # presente en WebSocket, ausente en REST
    trade_seller_order_id_col,      # presente en WebSocket, ausente en REST
    trade_date_col,                 # generado por BinPan
    trade_timestamp_col,
    trade_buyer_was_maker_col,
    trade_best_price_match_col,
]

# ============================================================================
# COLUMNAS AUXILIARES
# ============================================================================

# Columnas de tiempo en klines (strings legibles)
time_cols = [kline_open_time_col, kline_close_time_col]

# Columnas de timestamps en klines (int ms)
dts_time_cols = [kline_open_timestamp_col, kline_close_timestamp_col]

# Columnas para velas de reversal (construidas desde trades)
reversal_columns_order = [kline_open_col, kline_high_col, kline_low_col, kline_close_col, trade_quantity_col, trade_timestamp_col]

# ============================================================================
# POSTGRESQL
# ============================================================================

# Columnas de klines en PostgreSQL (sin Ignore, con First/Last TradeId)
postgresql_candles_ordered_cols = [
    kline_open_time_col,
    kline_open_col,
    kline_high_col,
    kline_low_col,
    kline_close_col,
    kline_volume_col,
    kline_close_time_col,
    kline_quote_volume_col,
    kline_trades_col,
    kline_taker_buy_base_volume_col,
    kline_taker_buy_quote_volume_col,
    kline_open_timestamp_col,
    kline_close_timestamp_col,
    kline_first_trade_id_col,
    kline_last_trade_id_col,
]

# Columnas de atomic trades en PostgreSQL (incluye buyer/seller order IDs)
postgresql_trades_cols_order = [
    trade_trade_id_col,
    trade_price_col,
    trade_quantity_col,
    trade_quote_quantity_col,
    trade_buyer_order_id_col,
    trade_seller_order_id_col,
    trade_date_col,
    trade_timestamp_col,
    trade_buyer_was_maker_col,
]

# Columnas de aggTrades en PostgreSQL (sin Best price match)
postgresql_agg_trade_cols_order = [
    agg_trade_id_col,
    agg_trade_price_col,
    agg_quantity_col,
    agg_first_trade_id_col,
    agg_last_trade_id_col,
    agg_date_col,
    agg_time_col,
    agg_buyer_was_maker_col,
]

postgresql_presentation_type_columns_dict = {
    'kline': postgresql_candles_ordered_cols,
    'trade': postgresql_trades_cols_order,
    'aggTrade': postgresql_agg_trade_cols_order,
    'aggtrade': postgresql_agg_trade_cols_order,
}

# ============================================================================
# POSTGRESQL -> BINPAN (mapeo de nombres de columna)
# ============================================================================

postgresql2binpan_map_dict = {
    "kline": {
        "time": kline_open_time_col,
        "open": kline_open_col,
        "high": kline_high_col,
        "low": kline_low_col,
        "close": kline_close_col,
        "volume": kline_volume_col,
        "close_timestamp": kline_close_timestamp_col,
        "quote_volume": kline_quote_volume_col,
        "trades": kline_trades_col,
        "taker_buy_base_volume": kline_taker_buy_base_volume_col,
        "taker_buy_quote_volume": kline_taker_buy_quote_volume_col,
        "first_trade_id": kline_first_trade_id_col,
        "last_trade_id": kline_last_trade_id_col,
        "open_timestamp": kline_open_timestamp_col,
        "ignore": kline_ignore_col,
    },
    "trade": {
        "trade_id": trade_trade_id_col,
        "price": trade_price_col,
        "quantity": trade_quantity_col,
        "quote_quantity": trade_quote_quantity_col,
        "date": trade_date_col,
        "buyer_order_id": trade_buyer_order_id_col,
        "seller_order_id": trade_seller_order_id_col,
        "time": trade_timestamp_col,
        "buyer_was_maker": trade_buyer_was_maker_col,
        "best_price_match": trade_best_price_match_col,
        "seller_taker": trade_buyer_was_maker_col,
    },
    "aggTrade": {
        "trade_id": agg_trade_id_col,
        "price": agg_trade_price_col,
        "quantity": agg_quantity_col,
        "first_trade_id": agg_first_trade_id_col,
        "last_trade_id": agg_last_trade_id_col,
        "date": agg_date_col,
        "time": agg_time_col,
        "buyer_was_maker": agg_buyer_was_maker_col,
        "best_price_match": agg_best_price_match_col,
        "seller_taker": trade_buyer_was_maker_col,
    },
    "aggtrade": {
        "trade_id": agg_trade_id_col,
        "price": agg_trade_price_col,
        "quantity": agg_quantity_col,
        "first_trade_id": agg_first_trade_id_col,
        "last_trade_id": agg_last_trade_id_col,
        "date": agg_date_col,
        "time": agg_time_col,
        "buyer_was_maker": agg_buyer_was_maker_col,
        "best_price_match": agg_best_price_match_col,
        "seller_taker": trade_buyer_was_maker_col,
    },
}
