import pandas as pd

from objects.timestamps import Timestamp
from objects.timeframes import Timeframe
from typing import Union, List
from datetime import datetime
from handlers.standards import (atomic_trades_api_map_columns, postgresql2binpan_map_dict, agg_trades_columns_from_binance)


class Trades(Timeframe):  # Fetcher objeto para trincar de api, postgresql, csv, etc
    def __init__(self,
                 start: Union[str, int, datetime, Timestamp, None],
                 end: Union[str, int, datetime, Timestamp, None],
                 trades: Union[List[dict], pd.DataFrame],
                 trade_type: str = 'trade',
                 origin: str = 'binance_api',
                 ):

        super().__init__(start=start, end=end, tick_interval=None, closed=False)

        assert trade_type in ['trade', 'aggtrade', 'aggTrade'], f"Trades trade type must be 'trade', 'aggtrade' or 'aggTrade' not {trade_type}"
        assert origin in ['binance_api', 'csv', 'postgresql'], f"Trades origin must be 'binance_api', 'csv' or 'postgresql' not {origin}"

        self.trades = pd.DataFrame(trades) if isinstance(trades, list) else trades
        self.trade_type = trade_type
        self.origin = origin

        self.columns = self.set_columns()

    def set_columns(self):
        """
        Set the columns of the trades dataframe maps for fetched data conversion.
        """
        if self.trade_type == 'trade':
            if self.origin == 'binance_api':
                return atomic_trades_api_map_columns
            elif self.origin == 'csv':
                return None
            elif self.origin == 'postgresql':
                return postgresql2binpan_map_dict[self.trade_type]

        elif self.trade_type in ['aggtrade', 'aggTrade']:
            if self.origin == 'binance_api':
                return agg_trades_columns_from_binance
            elif self.origin == 'csv':
                return None
            elif self.origin == 'postgresql':
                return postgresql2binpan_map_dict[self.trade_type]

    def add_trade(self, trade):
        # verifica si existe el trade en el dataframe e informa, si no lo agrega
        self.trades.append(trade)

    def __str__(self):
        return str(self.trades)
