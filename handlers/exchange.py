import pandas as pd
import numpy as np
from datetime import datetime

from .market import get_prices_dic
from .quest import api_raw_get, api_raw_signed_get, check_weight
from .logs import Logs

base_url = 'https://api.binance.com'

stablecoins = ['PAX', 'TUSD', 'USDC', 'USDS', 'USDT', 'BUSD', 'DAI', 'UST', 'USDP', 'TRIBE', 'UST']

float_api_items = ['price', 'origQty', 'executedQty', 'cummulativeQuoteQty', 'stopLimitPrice', 'stopPrice', 'commission', 'qty',
                   'origQuoteOrderQty', 'makerCommission', 'takerCommission']
int_api_items = ['orderId', 'orderListId', 'transactTime', 'tradeId', 'transactionTime', 'updateTime', 'time']

exchange_logger = Logs(filename='./logs/exchange_logger.log', name='exchange_logger', info_level='INFO')


#########################
# exchange general data #
#########################


def get_exchange_info() -> dict:
    """
    Returns general exchange info from /api/v3/exchangeInfo endpoint.

    :return dict: dict_keys(['timezone', 'serverTime', 'rateLimits', 'exchangeFilters', 'symbols'])

    """
    endpoint = '/api/v3/exchangeInfo'
    return api_raw_get(endpoint=endpoint,
                       weight=10)


def get_info_dic() -> dict:
    """
    Get the dictionary of each symbol with its information from the exchange.

    :return dict: Returns info for each symbol as keys in the dict.
    """
    return {k['symbol']: k for k in get_exchange_info()['symbols']}


###########
# account #
###########


def get_account_status() -> dict:
    """
    Fetch account status detail.

    For Machine Learning limits, restrictions will be applied to account. If a user has been restricted by the ML system, they may
    check the reason and the duration by using the [/sapi/v1/account/status] endpoint

    :return dict: Example { "data": "Normal" }

    """
    endpoint = '/sapi/v1/account/status'
    return api_raw_signed_get(endpoint=endpoint,
                              weight=1)


#################
# weight limits #
#################


def get_exchange_limits(info_dict: dict = None) -> dict:
    """
    Binance manage several limits: RAW_REQUESTS, REQUEST_WEIGHT, and ORDERS rate limits.

    The headers for those limits, I assume that are:
    - RAW_REQUESTS: x-mbx-used-weight. Is cross all the api calls.
    - REQUEST_WEIGHT: Example: x-mbx-order-count-10s. Time interval limited requests.
    - ORDERS: Example: x-mbx-order-count-10s. Rate limit for orders.
    - X-SAPI-USED-IP-WEIGHT-1M: For sapi endpoint requests.

    Example response:

        {'X-SAPI-USED-IP-WEIGHT-1M': 1200,
         'x-mbx-order-count-10s': 50,
         'x-mbx-order-count-1d': 160000,
         'x-mbx-used-weight': 6100,
         'x-mbx-used-weight-1m': 1200}

    :return dict:
    """

    if not info_dict:
        info_dict = get_exchange_info()

    limits = info_dict['rateLimits']

    limits_dict = {}

    for i in limits:
        if i['rateLimitType'].upper() == 'ORDERS':
            k1 = f"x-mbx-order-count-{i['intervalNum']}{i['interval'][0].lower()}"
            k2 = f"x-mbx-order-count-{i['intervalNum']}{i['interval'][0].lower()}"

        elif i['rateLimitType'].upper() == 'REQUEST_WEIGHT':
            k1 = f"x-mbx-used-weight-{i['intervalNum']}{i['interval'][0].lower()}"
            k2 = f"X-SAPI-USED-IP-WEIGHT-{i['intervalNum']}{i['interval'][0].upper()}"

        elif i['rateLimitType'].upper() == 'RAW_REQUESTS':
            k1 = "x-mbx-used-weight"
            k2 = "x-mbx-used-weight"
        else:
            raise Exception("BinPan Rate Limit not parsed")

        v = i['limit']

        limits_dict[k1] = v
        limits_dict[k2] = v

    return limits_dict


##################
# symbol filters #
##################


def flatten_filter(filters: list) -> dict:
    ret = {}
    for f in filters:
        head = f['filterType']
        for k, v in f.items():
            if k != 'filterType':
                new_key = f"{head}_{k}"
                ret.update({new_key: v})
    return ret


def get_symbols_filters(info_dic: dict = None) -> dict:
    if not info_dic:
        info_dic = get_info_dic()
    return {k: flatten_filter(v['filters']) for k, v in info_dic.items()}


def get_filters(symbol: str, info_dic: dict = None) -> dict:
    if not info_dic:
        info_dic = get_info_dic()
    return {k: flatten_filter(v['filters']) for k, v in info_dic.items() if k == symbol}


def filter_tradeable(info_dic: dict) -> dict:
    return {k: v for k, v in info_dic.items() if v['status'].upper() == 'TRADING'}


def filter_spot(info_dic: dict) -> dict:
    return {k: v for k, v in info_dic.items() if 'SPOT' in v['permissions'] or 'spot' in v['permissions']}


def filter_margin(info_dic: dict) -> dict:
    return {k: v for k, v in info_dic.items() if 'MARGIN' in v['permissions'] or 'margin' in v['permissions']}


def filter_leveraged_tokens(info_dic: dict) -> dict:
    return {k: v for k, v in info_dic.items() if all(lev not in k for lev in ['UP', 'DOWN', 'BULL', 'BEAR'])}


def filter_not_legal(legal: list, info: dict, bases: dict, quotes: dict) -> dict:
    return {k: v for k, v in info.items() if (not bases[k] in legal) and (not quotes[k] in legal)}


#################
# Exchange Data #
#################


def get_precision(info_dic: dict = None) -> dict:
    if not info_dic:
        info_dic = get_info_dic()
    return {k: {'baseAssetPrecision': v['baseAssetPrecision'],
                'quoteAssetPrecision': v['quoteAssetPrecision'],
                'baseCommissionPrecision': v['baseCommissionPrecision'],
                'quoteCommissionPrecision': v['quoteCommissionPrecision']}
            for k, v in info_dic.items()}


def get_orderTypes_and_permissions(info_dic: dict = None) -> dict:
    if not info_dic:
        info_dic = get_info_dic()
    return {k: {'orderTypes': v['orderTypes'], 'permissions': v['permissions']} for k, v in info_dic.items()}


def get_fees_dict(symbol: str = None) -> dict:
    """
    Returns fees for a symbol or for every symbol if not passed.

    :param str symbol: Optional to request just one symbol instead of all.
    :return dict: A dict with maker and taker fees.
    """
    endpoint = '/sapi/v1/asset/tradeFee'
    if symbol:
        symbol = symbol.upper()
    ret = api_raw_signed_get(endpoint,
                             params={'symbol': symbol},
                             weight=1)
    return {i['symbol']: {'makerCommission': float(i['makerCommission']),
                          'takerCommission': float(i['takerCommission'])} for i in ret}


def get_fees(symbol: str = None) -> pd.DataFrame:
    """
    Returns fees for a symbol or for every symbol if not passed.

    :param str symbol: Optional to request just one symbol instead of all.
    :return pd.DataFrame: A pandas dataframe with all the fees applied each symbol.
    """
    ret = get_fees_dict(symbol=symbol)
    return pd.DataFrame(ret).transpose()


def get_system_status():
    """
    Fetch system status.

    Weight(IP): 1

    :return dict: 
        { 
           "status": 0,              // 0: normal，1：system maintenance
            "msg": "normal"           // "normal", "system_maintenance"
        }
    """
    return api_raw_get(endpoint='/sapi/v1/system/status',
                       weight=1)['msg']


def get_coins_info():
    """
    Get information of coins (available for deposit and withdraw) for user.

    GET /sapi/v1/capital/config/getall (HMAC SHA256)

    Weight(IP): 10

    :return pd.DataFrame, pd.DataFrame:
    """

    ret = api_raw_signed_get(endpoint='/sapi/v1/capital/config/getall',
                             weight=10)
    networks = []
    coins = []

    for i in ret:
        network_list = i.pop('networkList', None)
        if network_list:
            networks += network_list
        coins.append(i)
    # networks = list(set(networks))

    coins_df = pd.DataFrame(coins).drop_duplicates().set_index('coin')
    for col in coins_df.columns:
        coins_df[col] = pd.to_numeric(arg=coins_df[col], downcast='integer', errors='ignore')

    networks_df = pd.DataFrame(networks).drop_duplicates().set_index('coin')
    for col in networks_df.columns:
        networks_df[col] = pd.to_numeric(arg=networks_df[col], downcast='integer', errors='ignore')

    return coins_df.sort_index(), networks_df.sort_index()


def get_coins_info_list(coin: str = None) -> list:
    """
    Trae todas las monedas si no se especifica una.

    Devuelve una lista de diccionarios, uno por cada moneda"""

    endpoint = '/sapi/v1/capital/config/getall'
    check_weight(weight=10, endpoint=endpoint)
    ret = api_raw_signed_get(endpoint=endpoint)
    if not coin:
        return ret
    else:
        return [c for c in ret if c['coin'].upper() == coin.upper()]


def get_coins_info_dic(coin: str = None) -> dict:
    coins_data_list = get_coins_info_list(coin=coin)
    return {c['coin']: c for c in coins_data_list}


def get_legal_coins(coins_dic: dict = None) -> list:
    """Trae las monedas que contienen isLegalMoney=true"""
    if not coins_dic:
        coins_dic = get_coins_info_dic()
    return [coin for coin, data in coins_dic.items() if data['isLegalMoney'] is True]


#######################
# Exchange Statistics #
#######################

# TODO: traido de cache, posible borrado
# def get_exchange_info() -> dict:
#     """Returns dict_keys(['timezone', 'serverTime', 'rateLimits', 'exchangeFilters', 'symbols'])"""
#     check_weight(10)
#     endpoint = '/api/v3/exchangeInfo'
#     return api_raw_get(url=endpoint)
#
#
# def get_info_dic() -> dict:
#     """Obtiene el diccionario de cada symbol con su información del exchange"""
#     return {k['symbol']: k for k in get_exchange_info()['symbols']}


def get_quotes_dic(info_dic: dict = None) -> dict:
    if not info_dic:
        info_dic = get_info_dic()
    return {k: v['quoteAsset'] for k, v in info_dic.items()}


def get_bases_dic(info_dic: dict = None) -> dict:
    if not info_dic:
        info_dic = get_info_dic()
    return {k: v['baseAsset'] for k, v in info_dic.items()}


# TODO: traido de cache, posible borrado
# def get_exchange_dicts() -> tuple:
#     """Obtiene los diccionarios básicos de símbolos del exchange para operar."""
#     info_dict = get_info_dic()
#     quotes_dic = get_quotes_dic(info_dic=info_dict)
#     bases_dic = get_bases_dic(info_dic=info_dict)
#     symbols_filters = get_filters(info_dict)
#     return info_dict, quotes_dic, bases_dic, symbols_filters

#
# def get_prices_dic() -> dict:
#     check_weight(2)
#     endpoint = '/api/v3/ticker/price'
#     ret = get_response(url=endpoint)
#     return {d['symbol']: float(d['price']) for d in ret}


def exchange_status(tradeable=True,
                    spot_required=True,
                    margin_required=True,
                    drop_legal=True,
                    filter_leveraged=True,
                    info_dic: dict = None,
                    symbol_filters: dict = None) -> tuple:
    # get symbols dict from server with some operational_filters
    if not info_dic:
        info_dic = get_info_dic()

    bases_dic = {k: v['baseAsset'] for k, v in info_dic.items()}
    quotes_dic = {k: v['quoteAsset'] for k, v in info_dic.items()}

    filtered_info = info_dic

    if filter_leveraged:
        filtered_info = filter_leveraged_tokens(info_dic)
    if tradeable:  # elimina los no tradeables
        filtered_info = filter_tradeable(filtered_info)
    if spot_required:
        filtered_info = filter_spot(filtered_info)
    if margin_required:
        filtered_info = filter_margin(filtered_info)

    legal_coins = get_legal_coins()
    not_legal_pairs = filter_not_legal(legal_coins, info_dic, bases_dic, quotes_dic)
    if drop_legal:
        filtered_info = filter_not_legal(legal_coins, filtered_info, bases_dic, quotes_dic)

    if not symbol_filters:
        symbol_filters = get_symbols_filters(info_dic=info_dic)

    filtered_pairs = filtered_info.keys()
    return bases_dic, quotes_dic, legal_coins, not_legal_pairs, symbol_filters, filtered_pairs


def get_24h_statistics(symbol=None) -> dict:  # 24h rolling window
    endpoint = '/api/v3/ticker/24hr?'
    if symbol:
        check_weight(endpoint=endpoint, weight=1)
        return api_raw_get(endpoint=endpoint, params={'symbol': symbol}, weight=1)
    else:
        check_weight(endpoint=endpoint, weight=40)
        return api_raw_get(endpoint=endpoint, weight=40)


def not_iterative_coin_conversion(coin: str = 'ADA',
                                  prices: dict = None,
                                  try_coin: str = 'BTC',
                                  coin_qty: float = 1) -> float or None:
    if not prices:
        prices = get_prices_dic()
    if coin + try_coin in prices.keys():
        price = prices[coin + try_coin]
        try_symbol = f"{try_coin}USDT"
        return price * coin_qty * prices[try_symbol]
    elif try_coin + coin in prices.keys():
        price = 1 / prices[try_coin + coin]
        try_symbol = f"USDT{try_coin}"
        try:
            return price * coin_qty * prices[try_symbol]
        except KeyError:
            return None
    else:
        return None


def convert_to_other_coin(coin: str = 'BTC', convert_to: str = 'USDT', coin_qty: float = 1,
                          prices: dict = None) -> float:
    """Valora en la moneda indicada en convert la moneda pasada por su cantidad. S"""
    coin = coin.upper()
    convert_to = convert_to.upper()

    if coin == convert_to:
        return coin_qty

    if not prices:
        prices = get_prices_dic()

    symbol_a = coin + convert_to
    symbol_b = convert_to + coin

    if symbol_a in prices.keys():
        # print(f"{coin} Selected {symbol_a} for ", coin)
        return coin_qty * prices[symbol_a]

    elif symbol_b in prices.keys():
        # print(f"{coin} Selected {symbol_b} for ", coin)
        return coin_qty * (1 / prices[symbol_b])
    else:
        # try using btc
        # try:
        ret1 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='BTC', coin_qty=coin_qty)
        if ret1:
            return ret1
        ret2 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='BUSD', coin_qty=coin_qty)
        if ret2:
            return ret2
        ret3 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='BNB', coin_qty=coin_qty)
        if ret3:
            return ret3
        ret4 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='ETH', coin_qty=coin_qty)
        if ret4:
            return ret4
        ret5 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='TUSD', coin_qty=coin_qty)
        if ret5:
            return ret5
        ret6 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='USDC', coin_qty=coin_qty)
        if ret6:
            return ret6
        else:
            return np.nan


def convert_symbol_base_to_other_coin(symbol_to_convert_base: str = 'ETHBTC',
                                      base_qty: float = 1,
                                      convert_to: str = 'USDT',
                                      prices: dict = None,
                                      info_dic: dict = None) -> float:
    if not info_dic:
        info_dic = get_info_dic()
    bases_dic = {k: v['baseAsset'] for k, v in info_dic.items()}

    base = bases_dic[symbol_to_convert_base]
    if not prices:
        prices = get_prices_dic()
    ret = convert_to_other_coin(coin=base, convert_to=convert_to, coin_qty=base_qty, prices=prices)
    if type(ret) == int or type(ret) == float:
        return ret
    else:
        raise Exception('BinPan Error: convert_symbol_base_to_other_coin breakpoint')


def convert_utc_milliseconds(ms) -> str:
    seconds = int(ms) / 1000
    return str(datetime.utcfromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f'))


def statistics_24h(tradeable=True,
                   spot_required=True,
                   margin_required=False,
                   drop_legal=True,
                   filter_leveraged=True,
                   info_dic=None,
                   stablecoin_value='BUSD',
                   sort_by: str = 'priceChangePercent') -> pd.DataFrame:
    """Genera un dataframe con los filtros a aplicar con las estadísticas de las últimas 24 horas. Opcionalmente,
    se puede generar la columna de convertir el volumen a USDT

    TODO: INCORPORAR ESTO A LA MATRIZ PRINCIPAL

    """

    if not info_dic:
        info_dic = get_info_dic()

    bases_dic, quotes_dic, _, not_legal_pairs, _, filtered_pairs = exchange_status(tradeable=tradeable,
                                                                                   spot_required=spot_required,
                                                                                   margin_required=margin_required,
                                                                                   drop_legal=drop_legal,
                                                                                   filter_leveraged=filter_leveraged,
                                                                                   info_dic=info_dic)
    stats = get_24h_statistics()

    df = pd.DataFrame(stats)

    # filter just coins not legal and with spot and tradeable
    df = df[df['symbol'].isin(filtered_pairs)]

    df = df.apply(pd.to_numeric, errors='ignore')

    df['openTime'] = df['openTime'].apply(lambda x: convert_utc_milliseconds(x))
    df['closeTime'] = df['closeTime'].apply(lambda x: convert_utc_milliseconds(x))

    df = df.set_index('symbol', drop=False)

    # df['filter_passed'] = df['symbol'].apply(lambda x: True if x in filtered_pairs else False)
    df['is_legal'] = df['symbol'].apply(lambda x: True if x not in not_legal_pairs else False)

    df['base'] = df['symbol'].apply(lambda x: bases_dic[x])
    df['quote'] = df['symbol'].apply(lambda x: quotes_dic[x])

    if stablecoin_value:
        prices = get_prices_dic()
        # info_dic = {k['symbol']: k for k in market.get_exchange_info()['symbols']}
        # bases_dic = market.get_bases_dic(info_dic=info_dic)
        # quotes_dic = market.get_quotes_dic(info_dic=info_dic)

        stable_coin_value_name = f"{stablecoin_value}_value"

        for pair in prices.keys():
            usdt_val = convert_symbol_base_to_other_coin(symbol_to_convert_base=pair,
                                                         base_qty=1,
                                                         convert_to=stablecoin_value,
                                                         prices=prices,
                                                         info_dic=info_dic)
            df.loc[df['symbol'] == pair, stable_coin_value_name] = usdt_val
        stable_coin_volumen_name = f"{stablecoin_value}_volume"
        df[stable_coin_volumen_name] = df[stable_coin_value_name] * df['volume']
        return df.sort_values(stable_coin_volumen_name, ascending=False)
    return df.sort_values(sort_by, ascending=False)


def get_top_gainers(info_dic: dict = None,
                    tradeable=True,
                    spot_required=True,
                    margin_required=False,
                    drop_legal=True,
                    filter_leveraged=True,
                    top_gainers_qty: int = None,
                    my_quote: str = 'BUSD',
                    drop_stable_pairs: bool = True,
                    sort_by_column: str = 'priceChangePercent',  # priceChangePercent
                    ) -> pd.DataFrame:
    """
    TODO: INCORPORAR ESTO A LA MATRIZ PRINCIPAL

    """
    if not info_dic:
        info_dic = get_info_dic()

    top_gainers = statistics_24h(tradeable=tradeable,
                                 spot_required=spot_required,
                                 margin_required=margin_required,
                                 drop_legal=drop_legal,
                                 info_dic=info_dic,
                                 filter_leveraged=filter_leveraged,
                                 stablecoin_value=my_quote)

    top_gainers = top_gainers.loc[top_gainers['quote'] == my_quote]

    if drop_stable_pairs:
        stable_pairs = [f"{s}{my_quote}" for s in stablecoins]
        stable_pairs += [f"{my_quote}{s}" for s in stablecoins]

        top_gainers = top_gainers.loc[~top_gainers['symbol'].isin(stable_pairs)]

    top_gainers = top_gainers[['priceChangePercent', 'volume']]

    if top_gainers_qty:
        return top_gainers.sort_values(sort_by_column, ascending=False).head(top_gainers_qty)
    else:
        return top_gainers.sort_values(sort_by_column, ascending=False)
