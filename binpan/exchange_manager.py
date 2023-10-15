import pandas as pd
import handlers.files
from handlers.exchange import (get_info_dic, get_coins_info_dic, get_bases_dic, get_quotes_dic, get_leveraged_coins,
                               get_leveraged_symbols, get_fees, get_symbols_filters, get_system_status, get_coins_and_networks_info)


class Exchange(object):
    """
    Exchange data.

    Exchange data collected in class variables:

    - **my_exchange_instance.info_dic**: A dictionary with all raw symbols info each.
    - **my_exchange_instance.coins_dic**: A dictionary with all coins info.
    - **my_exchange_instance.bases**: A dictionary with all bases for all symbols.
    - **my_exchange_instance.quotes**: A dictionary with all quotes for all symbols.
    - **my_exchange_instance.leveraged**: A list with all leveraged coins.
    - **my_exchange_instance.leveraged_symbols**: A list with all leveraged symbols.
    - **my_exchange_instance.fees**: dataframe with fees applied to the user requesting for every symbol.
    - **my_exchange_instance.filters**: A dictionary with all trading filters detailed with all symbols.
    - **my_exchange_instance.status**: API status can be normal o under maintenance.
    - **my_exchange_instance.coins**: A dataframe with all the coin's data.
    - **my_exchange_instance.networks**: A dataframe with info about every coin and its blockchain networks info.
    - **my_exchange_instance.coins_list**: A list with all the coin's names.
    - **my_exchange_instance.symbols**: A list with all the symbols names.
    - **my_exchange_instance.df**: A dataframe with all the symbols info.
    - **my_exchange_instance.order_types**: Dataframe with each symbol order types.

    """

    def __init__(self):
        self.api_key, self.api_secret = handlers.files.get_encoded_secrets()

        self.info_dic = get_info_dic()
        self.coins_dic = get_coins_info_dic(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.bases = get_bases_dic(info_dic=self.info_dic)
        self.quotes = get_quotes_dic(info_dic=self.info_dic)
        self.leveraged = get_leveraged_coins(coins_dic=self.coins_dic, decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.leveraged_symbols = get_leveraged_symbols(info_dic=self.info_dic, leveraged_coins=self.leveraged, decimal_mode=False,
                                                       api_key=self.api_key, api_secret=self.api_secret)

        self.fees = get_fees(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.filters = get_symbols_filters(info_dic=self.info_dic)
        self.status = get_system_status()
        self.coins, self.networks = get_coins_and_networks_info(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.coins_list = list(self.coins.index)

        self.symbols = self.get_symbols()
        self.df = self.get_df()
        self.order_types = self.get_order_types()

        # 24h things
        self.usdt_volume_24h = self.get_volume_24h()
        self.statistics_24h = self.get_statistics_24h()

    def __repr__(self):
        return str(self.df)

    def filter(self, symbol: str):
        """
        Returns exchange filters applied for orders with the selected symbol.

        :param str symbol:
        :return dict:
        """
        return self.filters[symbol.upper()]

    def fee(self, symbol: str):
        """
        Returns exchange fees applied for orders with the selected symbol.

        :param str symbol:
        :return pd.Series:
        """
        return self.fees.loc[symbol.upper()]

    def coin(self, coin: str):
        """
        Returns coin exchange info in a pandas serie.

        :param str coin:
        :return pd.Series:
        """
        return self.coins.loc[coin.upper()]

    def network(self, coin: str):
        """
        Returns a dataframe with all exchange networks of one specified coin or every coin.

        :param str coin:
        :return pd.Series:
        """
        return self.networks.loc[coin.upper()]

    def update_info(self, symbol: str = None):
        """
        Updates from API and returns a dict with all merged exchange info about a symbol.

        :param str symbol:
        :return dict:
        """
        self.info_dic = get_info_dic()
        self.filters = get_symbols_filters(info_dic=self.info_dic)
        self.bases = get_bases_dic(info_dic=self.info_dic)
        self.quotes = get_quotes_dic(info_dic=self.info_dic)
        self.fees = get_fees(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.status = get_system_status()
        self.coins, self.networks = get_coins_and_networks_info(decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.symbols = self.get_symbols()
        self.df = self.get_df()
        self.order_types = self.get_order_types()

        if symbol:
            return self.info_dic[symbol.upper()]
        return self.info_dic

    def get_symbols(self, coin: str = None, base: bool = True, quote: bool = True):
        """
        Return list of symbols for a coin. Can be selected symbols where it is base, or quote, or both.

        :param str coin: An existing binance coin.
        :param bool base: Activate return of symbols where coin is base.
        :param bool quote: Activate return of symbols where coin is quote.
        :return list: List of symbols where it is base, quote or both.
        """
        if not coin:
            self.symbols = list(self.info_dic.keys())
            return self.symbols

        else:
            ret = []
            if base:
                ret += [i for i in self.symbols if self.bases[i] == coin.upper()]
            if quote:
                ret += [i for i in self.symbols if self.quotes[i] == coin.upper()]
            return ret

    def get_df(self):
        """
        Extended symbols dataframe with exchange info about trading permissions, trading or blocked symbol, order types, margin allowed, etc

        :return pd.DataFrame: An exchange dataframe with all symbols data.

        """
        df = pd.DataFrame(self.info_dic.values()).set_index('symbol', drop=True)
        per_df = pd.DataFrame(data=df['permissions'].tolist(), index=df.index)
        per_df.columns = per_df.value_counts().index[0]
        self.df = pd.concat([df.drop('permissions', axis=1), per_df.astype(bool)], axis=1)
        return self.df

    def get_order_types(self):
        """
        Returns a dataframe with order types for symbol.

        :return pd.DataFrame:
        """
        ord_df = pd.DataFrame(data=self.df['orderTypes'].tolist(), index=self.df.index)
        ord_df.columns = ord_df.value_counts().index[0]
        self.order_types = ord_df.astype(bool)
        return self.order_types

    def get_volume_24h(self,
                       quote: str = "USDT",
                       tradeable=True,
                       spot_required=True,
                       margin_required=True,
                       drop_legal=True,
                       filter_leveraged=True,
                       info_dic=None,
                       sort_by: str = None) -> pd.DataFrame:
        """
        Returns a dataframe with 24h busd volume for every symbol.

        :param quote: Optional quote to filter.
        :param tradeable: Optional filter to return only tradeable symbols.
        :param spot_required: Optional filter to return only spot tradeable symbols.
        :param margin_required: Optional filter to return only margin tradeable symbols.
        :param drop_legal: Optional filter to drop legal symbols.
        :param filter_leveraged: Optional filter to drop leveraged symbols.
        :param info_dic: Optional info dictionary to use.
        :param sort_by: Optional column to sort by.
        :return: A dataframe with 24h busd volume for every symbol.
        """
        quote = quote.upper()
        if not sort_by:
            sort_by = f'{quote}_volume'

        ret = handlers.exchange.statistics_24h(decimal_mode=True,
                                               api_key=self.api_key,
                                               api_secret=self.api_secret,
                                               info_dic=info_dic,
                                               tradeable=tradeable,
                                               spot_required=spot_required,
                                               margin_required=margin_required,
                                               drop_legal=drop_legal,
                                               filter_leveraged=filter_leveraged,
                                               stablecoin_value=quote).sort_values(sort_by, ascending=False)
        columns = ['symbol', sort_by, 'openPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'weightedAvgPrice']
        ret = ret[columns + [c for c in ret.columns if c not in columns]]
        if quote:
            self.usdt_volume_24h = ret.loc[ret['quote'] == quote.upper()]
        else:
            self.usdt_volume_24h = ret
        return self.usdt_volume_24h


    def get_usdt_volume_24h(self, quote=None) -> pd.DataFrame:
        """
        Returns a dataframe with 24h busd volume for every symbol.

        :param quote: Optional quote to filter.
        :return: A dataframe with 24h busd volume for every symbol.
        """
        ret = handlers.exchange.statistics_24h(decimal_mode=True,
                                               api_key=self.api_key,
                                               api_secret=self.api_secret,
                                               stablecoin_value="USDT").sort_values('USDT_volume', ascending=False)

        columns = ['symbol', 'USDT_volume', 'openPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'weightedAvgPrice']
        ret = ret[columns + [c for c in ret.columns if c not in columns]]

        if quote:
            self.busd_volume_24h = ret.loc[ret['quote'] == quote.upper()]
        else:
            self.busd_volume_24h = ret
        return self.busd_volume_24h

    def get_statistics_24h(self, symbol: str = None, quote: str = None) -> pd.DataFrame:
        """
        Returns a dataframe with 24h statistics for every symbol.

        :param symbol: Optional symbol to filter.
        :param quote: Optional quote to filter.
        :return: A dataframe with 24h statistics for every symbol.
        """
        ret = (handlers.exchange.statistics_24h(decimal_mode=True,
                                                api_key=self.api_key,
                                                api_secret=self.api_secret).sort_values('priceChangePercent', ascending=False))

        columns = ['symbol', 'priceChangePercent', 'openPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'weightedAvgPrice']
        ret = ret[columns + [c for c in ret.columns if c not in columns]]

        if symbol:
            ret = ret.loc[ret['symbol'] == symbol.upper()]
        if quote:
            ret = ret.loc[ret['quote'] == quote.upper()]
        self.statistics_24h = ret
        return ret
