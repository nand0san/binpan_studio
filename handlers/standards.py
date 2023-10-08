stream_uniqueness_id_in_timescale = {'aggTrade': 'trade_id',
                                     'trade': 'trade_id',
                                     'kline': 'time',
                                     'statistics': 'time'}

########################
# BINPAN PRESENTATIONS #
########################

# klines columns names

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

binance_api_candles_cols = [kline_open_time_col,
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
                            kline_ignore_col,
                            kline_open_timestamp_col,
                            kline_close_timestamp_col]

postgresql_candles_ordered_cols = [kline_open_time_col,
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
                                   kline_last_trade_id_col]

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
agg_time_col = trade_time_col
agg_buyer_was_maker_col = trade_buyer_was_maker_col
agg_best_price_match_col = trade_best_price_match_col

binance_api_agg_trade_cols = [agg_trade_id_col, agg_trade_price_col, agg_quantity_col, agg_first_trade_id_col,
                              agg_last_trade_id_col, agg_date_col, agg_time_col, agg_buyer_was_maker_col,
                              agg_best_price_match_col]
postgresql_agg_trade_cols = None

binpan_type_columns_dict = {'kline': binance_api_candles_cols, 'trade': binance_api_trades_cols, 'aggTrade': binance_api_agg_trade_cols}

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

postgresql_response_cols_by_type = {'kline': [kline_open_time_col,
                                              kline_close_timestamp_col,
                                              kline_first_trade_id_col,
                                              kline_last_trade_id_col,
                                              kline_open_col,
                                              kline_close_col,
                                              kline_high_col,
                                              kline_low_col,
                                              kline_volume_col,
                                              kline_trades_col,
                                              kline_quote_volume_col,
                                              kline_taker_buy_base_volume_col,
                                              kline_taker_buy_quote_volume_col],
                                    'trade': [trade_trade_id_col,
                                              trade_price_col,
                                              trade_quantity_col,
                                              trade_quote_quantity_col,
                                              trade_buyer_order_id_col,
                                              trade_seller_order_id_col,
                                              trade_date_col,
                                              trade_buyer_was_maker_col,
                                              ],
                                    'aggTrade': None
                                    }
postgresql_presentation_type_columns_dict = {'kline': postgresql_candles_ordered_cols,
                                             'trade': postgresql_trades_cols,
                                             'aggTrade': postgresql_agg_trade_cols}

# FROM POSTGRESQL TO BINPAN

postgresql2binpan_renamer_dict = {"kline": {"time": kline_open_time_col,
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
                                            "ignore": kline_ignore_col},
                                  "trade": {"trade_id": trade_trade_id_col,
                                            "price": trade_price_col,
                                            "quantity": trade_quantity_col,
                                            "quote_quantity": trade_quote_quantity_col,
                                            "date": trade_date_col,
                                            "buyer_order_id": trade_buyer_order_id_col,
                                            "seller_order_id": trade_seller_order_id_col,
                                            "time": trade_time_col,
                                            "buyer_was_maker": trade_buyer_was_maker_col,
                                            "best_price_match": trade_best_price_match_col},
                                  "aggTrade": {"aggregate_trade_id": agg_trade_id_col,
                                               "price": agg_trade_price_col,
                                               "quantity": agg_quantity_col,
                                               "first_trade_id": agg_first_trade_id_col,
                                               "last_trade_id": agg_last_trade_id_col,
                                               "date": agg_date_col,
                                               "time": agg_time_col,
                                               "buyer_was_maker": agg_buyer_was_maker_col,
                                               "best_price_match": agg_best_price_match_col}
                                  }
