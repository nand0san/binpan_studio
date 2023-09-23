########################
# BINPAN PRESENTATIONS #
########################

# klines columns names

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

binance_api_candles_cols = [open_time_col,
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

postgresql_candles_cols = None

# presentation_columns = [open_time_col, open_col, high_col, low_col, close_col, close_time_col,
#                         volume_col, quote_volume_col, trades_col,
#                         taker_buy_base_volume_col, taker_buy_quote_volume_col, ignore_col, open_timestamp_col, close_timestamp_col]

# trade columns names
trade_trade_id_col = 'Trade Id'
trade_price_col = 'Price'
trade_quantity_col = 'Quantity'
trade_quote_quantity_col = 'Quote quantity'
trade_date_col = 'Date'
trade_buyer_order_id_col = 'Buyer Order Id'
trade_seller_order_id_col = 'Seller Order Id'
trade_time_col = 'Timestamp'
trade_buyer_was_maker_col = 'Buyer was maker'
trade_best_price_match_col = 'Best price match'

binance_api_trades_cols = [trade_trade_id_col, trade_price_col, trade_quantity_col, trade_quote_quantity_col,
                           trade_date_col, trade_time_col, trade_buyer_was_maker_col, trade_best_price_match_col]

postgresql_trades_cols = [trade_trade_id_col, trade_price_col, trade_quantity_col, trade_quote_quantity_col,
                          trade_buyer_order_id_col, trade_seller_order_id_col, trade_time_col,
                          trade_buyer_was_maker_col]
# agg trade columns names
agg_trade_id_col = 'Aggregate tradeId'
agg_trade_price_col = trade_price_col
agg_quantity_col = trade_quantity_col
agg_first_trade_id_col = 'First tradeId'
agg_last_trade_id_col = 'Last tradeId'
agg_date_col = trade_date_col
agg_timestamp_col = trade_time_col
agg_buyer_was_maker_col = trade_buyer_was_maker_col
agg_best_price_match_col = trade_best_price_match_col

binance_api_agg_trade_cols = [agg_trade_id_col, agg_trade_price_col, agg_quantity_col, agg_first_trade_id_col,
                              agg_last_trade_id_col, agg_date_col, agg_timestamp_col, agg_buyer_was_maker_col,
                              agg_best_price_match_col]
postgresql_agg_trade_cols = None

binpan_type_columns_dict = {'kline': binance_api_candles_cols, 'trade': binance_api_trades_cols, 'aggTrade': binance_api_agg_trade_cols}
postgresql_presentation_type_columns_dict = {'kline': postgresql_candles_cols, 'trade': postgresql_trades_cols,
                                             'aggTrade': postgresql_agg_trade_cols}

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

# market things
klines_columns = {"t": "Open time", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "T": "Close time",
                  "q": "Quote volume", "n": "Trades", "V": "Taker buy base volume", "Q": "Taker buy quote volume", "B": "Ignore"}

trades_columns = {'M': 'Best price match', 'm': 'Buyer was maker', 'T': 'Timestamp', 'l': 'Last tradeId', 'f': 'First tradeId',
                  'q': 'Quantity', 'p': 'Price', 'a': 'Aggregate tradeId'}

agg_trades_columns_from_binance = ['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Date', 'Timestamp',
                                   'Buyer was maker', 'Best price match']
agg_trades_columns_from_redis = ['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Date', 'Timestamp',
                                 'Buyer was maker', 'Best price match']
atomic_trades_columns_from_binance = ['Trade Id', 'Price', 'Quantity', 'Quote quantity', 'Date', 'Timestamp', 'Buyer was maker',
                                      'Best price match']
atomic_trades_columns_from_redis = ['Trade Id', 'Price', 'Quantity', 'Buyer Order Id', 'Seller Order Id', 'Date', 'Timestamp',
                                    'Buyer was maker', 'Best price match']

# postgresql things

# atomic trades columns in postgresql
#              {'id': 105502780,
#               'price': '63.30000000',
#               'qty': '0.47100000',
#               'quoteQty': '29.81430000',
#               'buyerOrderId': 22427634644 or NONE,  # EN LLAMADAS A api NO VIENEN
#               'sellerOrderId': 22427637651 or NONE,  # EN LLAMADAS A api NO VIENEN
#               'time': 1695467325926,
#               'isBuyerMaker': True,
#               'isBestMatch': True}

postgresql_response_cols_by_type = {'kline': None,
                                    'trade': [trade_trade_id_col, trade_price_col, trade_quantity_col, trade_quote_quantity_col,
                                              trade_buyer_order_id_col, trade_seller_order_id_col, trade_date_col,
                                              trade_buyer_was_maker_col, trade_best_price_match_col],
                                    'aggTrade': None
                                    }
