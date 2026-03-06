import pandas as pd
from .api.exchange_info import (get_info_dic, get_coins_info_dic, get_bases_dic, get_quotes_dic, get_leveraged_coins,
                               get_leveraged_symbols, get_fees, get_symbols_filters, get_system_status,
                               get_coins_and_networks_info, statistics_24h)


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

    Credentials are managed by panzer's CredentialManager (~/.panzer_creds).
    On first use, panzer will prompt for API key and secret if not already stored.
    """

    def __init__(self):
        self.info_dic = get_info_dic()
        self.coins_dic = get_coins_info_dic(decimal_mode=False)
        self.bases = get_bases_dic(info_dic=self.info_dic)
        self.quotes = get_quotes_dic(info_dic=self.info_dic)
        self.leveraged = get_leveraged_coins(coins_dic=self.coins_dic, decimal_mode=False)
        self.leveraged_symbols = get_leveraged_symbols(info_dic=self.info_dic, leveraged_coins=self.leveraged, decimal_mode=False)

        self.fees = get_fees(decimal_mode=False)
        self.filters = get_symbols_filters(info_dic=self.info_dic)
        self.status = get_system_status()
        self.coins, self.networks = get_coins_and_networks_info(decimal_mode=False)
        self.coins_list = list(self.coins.index)

        self.symbols = self.get_symbols()
        self.df = self.get_df()
        self.order_types = self.get_order_types()

        # 24h things
        self.usdt_volume_24h = self.get_volume_24h()
        self.statistics_24h = self.get_statistics_24h()
        self.busd_volume_24h = None

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
        self.fees = get_fees(decimal_mode=False)
        self.status = get_system_status()
        self.coins, self.networks = get_coins_and_networks_info(decimal_mode=False)
        self.symbols = self.get_symbols()
        self.df = self.get_df()
        self.order_types = self.get_order_types()

        if symbol:
            return self.info_dic[symbol.upper()]
        return self.info_dic

    def get_symbols(self, coin: str = None, base: bool = True, quote: bool = True):
        """
        Return list of all symbols for a coin. Can be selected symbols where it is base, or quote, or both.
        By default, returns all symbols where coin is base or quote but you can deactivate one of them.

        :param str coin: An existing binance coin.
        :param bool base: Activate return of symbols where coin is base. Default is True.
        :param bool quote: Activate return of symbols where coin is quote. Default is True.
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

        # Binance migró permissions → permissionSets (lista de listas). Filtrar TRD_GRP_* internos.
        perm_col = 'permissionSets' if 'permissionSets' in df.columns else 'permissions'

        def _flatten_perms(perm_data):
            if not perm_data:
                return set()
            if isinstance(perm_data[0], list):
                return {p for sublist in perm_data for p in sublist if not p.startswith('TRD_GRP_')}
            return {p for p in perm_data if not p.startswith('TRD_GRP_')}

        flat_perms = df[perm_col].apply(_flatten_perms)
        all_perms = set()
        for s in flat_perms:
            all_perms.update(s)

        per_df = pd.DataFrame(index=df.index)
        for perm in sorted(all_perms):
            per_df[perm] = flat_perms.apply(lambda x, p=perm: p in x)

        cols_to_drop = [c for c in ('permissions', 'permissionSets') if c in df.columns]
        self.df = pd.concat([df.drop(cols_to_drop, axis=1), per_df], axis=1)
        return self.df

    def get_order_types(self):
        """
        Returns a dataframe with order types for symbol.

        :return pd.DataFrame:
        """
        all_types = set()
        for types_list in self.df['orderTypes']:
            all_types.update(types_list)

        ord_df = pd.DataFrame(index=self.df.index)
        for ot in sorted(all_types):
            ord_df[ot] = self.df['orderTypes'].apply(lambda x, t=ot: t in x)

        self.order_types = ord_df
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
        Returns a dataframe with 24-hour USDT volume for each cryptocurrency symbol.

        :param quote: Optional string to filter by a specific quote currency (e.g., 'USDT').
        :param tradeable: Optional boolean to return only tradeable symbols.
        :param spot_required: Optional boolean to return only spot tradeable symbols.
        :param margin_required: Optional boolean to return only margin tradeable symbols.
        :param drop_legal: Optional boolean to exclude legal symbols.
        :param filter_leveraged: Optional boolean to exclude leveraged symbols.
        :param info_dic: Optional dictionary containing additional information.
        :param sort_by: Optional string to sort the dataframe by a specific column.
        :return: Pandas DataFrame with columns like 'symbol', 'USDT_volume', 'openPrice',
                 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'weightedAvgPrice',
                 'priceChange', 'priceChangePercent', 'lastPrice', etc., for each symbol.
        """
        quote = quote.upper()
        if not sort_by:
            sort_by = f'{quote}_volume'

        ret = statistics_24h(decimal_mode=True,
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
        ret = statistics_24h(decimal_mode=True,
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
        ret = statistics_24h(decimal_mode=True).sort_values('priceChangePercent', ascending=False)

        columns = ['symbol', 'priceChangePercent', 'openPrice', 'highPrice', 'lowPrice', 'volume', 'quoteVolume', 'weightedAvgPrice']
        ret = ret[columns + [c for c in ret.columns if c not in columns]]

        if symbol:
            ret = ret.loc[ret['symbol'] == symbol.upper()]
        if quote:
            ret = ret.loc[ret['quote'] == quote.upper()]
        self.statistics_24h = ret
        return ret
