########################
# BINPAN PRESENTATIONS #
########################


open_time_col = 'Open time'
open_col = 'Open'
high_col = 'High'
low_col = 'Low'
close_col = 'Close'
volume_col = 'Volume'
close_time_col = 'Close time'
quote_volume_col = 'Quote volume'
trades_col = 'Trades'
taker_buy_base_volume_col = 'Taker buy base volume'
taker_buy_quote_volume_col = 'Taker buy quote volume'
ignore_col = 'Ignore'
open_timestamp_col = 'Open timestamp'
close_timestamp_col = 'Close timestamp'

original_candles_cols = [open_time_col,
                         open_col,
                         high_col,
                         low_col,
                         close_col,
                         volume_col,
                         close_time_col,
                         quote_volume_col,
                         trades_col,
                         taker_buy_base_volume_col,
                         taker_buy_quote_volume_col,
                         ignore_col,
                         open_timestamp_col,
                         close_timestamp_col]
presentation_columns = [open_col, high_col, low_col, close_col, volume_col, quote_volume_col, trades_col, taker_buy_base_volume_col, taker_buy_quote_volume_col]

# BINANCE API FIELD AND COLUMN NAMES

agg_trades_columns = {'M': 'Best price match', 'm': 'Buyer was maker', 'T': 'Timestamp', 'l': 'Last tradeId',
                      'f': 'First tradeId', 'q': 'Quantity', 'p': 'Price', 'a': 'Aggregate tradeId'}

atomic_trades_columns = {'id': 'Trade Id', 'price': 'Price', 'qty': 'Quantity', 'quoteQty': 'Quote quantity',
                         'time': 'Timestamp', 'isBuyerMaker': 'Buyer was maker', 'isBestMatch': 'Best price match'}

# updated from binpan cache modificados para el parser
agg_trades_columns_redis = {'a': "Aggregate tradeId", 'p': 'Price', 'q': 'Quantity', 'f': "First tradeId", 'l': "Last tradeId",
                            'T': "Timestamp", 'm': "Buyer was maker", 'M': "Best price match"}

# updated from binpan cache modificados para el parser
atomic_trades_columns_redis = {'t': "Trade Id", 'p': 'Price', 'q': 'Quantity',  # quote qty missing in atomic trades websockets
                               'b': "Buyer Order Id", 'a': "Seller Order Id", 'T': "Timestamp", 'm': "Buyer was maker",
                               'M': "Best price match"}

reversal_columns = ['Open', 'High', 'Low', 'Close', 'Quantity', 'Timestamp']

time_cols = ['Open time', 'Close time']

dts_time_cols = ['Open timestamp', 'Close timestamp']
