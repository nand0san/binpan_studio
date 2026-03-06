import pandas as pd

from datetime import datetime
from kline_timestamp import KlineTimestamp
from .timeframes import Timeframe
from .standards import (atomic_trades_api_map_columns, postgresql2binpan_map_dict, agg_trades_columns_from_binance)


class Trades(Timeframe):
    def __init__(self,
                 symbol: str,
                 start: str | int | datetime | KlineTimestamp | None,
                 end: str | int | datetime | KlineTimestamp | None,
                 trades: list[dict] | pd.DataFrame,
                 trade_type: str = 'trade',
                 origin: str = 'binance_api',
                 timezone_IANA: str = 'Europe/Madrid',
                 tick_interval: str | None = None,
                 closed: bool = False):

        Timeframe.__init__(self,
                           start=start,
                           end=end,
                           timezone_IANA=timezone_IANA,
                           tick_interval=tick_interval,
                           closed=closed)

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
    # TODO: IMPLEMENTAR FUNCIONES PARA PARSEO DE DISTINTOS ORIGENES DE DATOS
    #       - binance api
    #       - csv
    #       - postgresql

    def to_csv(self, filepath: str):
        self.trades.to_csv(filepath)

