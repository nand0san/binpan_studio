import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal as dd

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

    Example:

       ..code-block::

            {
              "timezone": "UTC",
              "serverTime": 1565246363776,
              "rateLimits": [
                {
                  //These are defined in the `ENUM definitions` section under `Rate Limiters (rateLimitType)`.
                  //All limits are optional
                }
              ],
              "exchangeFilters": [
                //These are the defined filters in the `Filters` section.
                //All filters are optional.
              ],
              "symbols": [
                {
                  "symbol": "ETHBTC",
                  "status": "TRADING",
                  "baseAsset": "ETH",
                  "baseAssetPrecision": 8,
                  "quoteAsset": "BTC",
                  "quotePrecision": 8,
                  "quoteAssetPrecision": 8,
                  "orderTypes": [
                    "LIMIT",
                    "LIMIT_MAKER",
                    "MARKET",
                    "STOP_LOSS",
                    "STOP_LOSS_LIMIT",
                    "TAKE_PROFIT",
                    "TAKE_PROFIT_LIMIT"
                  ],
                  "icebergAllowed": true,
                  "ocoAllowed": true,
                  "quoteOrderQtyMarketAllowed": true,
                  "allowTrailingStop": false,
                  "cancelReplaceAllowed": false,
                  "isSpotTradingAllowed": true,
                  "isMarginTradingAllowed": true,
                  "filters": [
                    //These are defined in the Filters section.
                    //All filters are optional
                  ],
                  "permissions": [
                     "SPOT",
                     "MARGIN"
                  ]
                }
              ]
            }
    """
    endpoint = '/api/v3/exchangeInfo'
    return api_raw_get(endpoint=endpoint,
                       weight=10)


def get_info_dic() -> dict:
    """
    Get the dictionary of each symbol with its information from the exchange.

    :return dict: Returns info for each symbol as keys in the dict.

    Example:

    .. code-block::

       from handlers import exchange

       info_dic = exchange.get_info_dic()

       info_dic['ETHUSDT']

        {'symbol': 'ETHUSDT',
         'status': 'TRADING',
         'baseAsset': 'ETH',
         'baseAssetPrecision': 8,
         'quoteAsset': 'USDT',
         'quotePrecision': 8,
         'quoteAssetPrecision': 8,
         'baseCommissionPrecision': 8,
         'quoteCommissionPrecision': 8,
         'orderTypes': ['LIMIT',
          'LIMIT_MAKER',
          'MARKET',
          'STOP_LOSS_LIMIT',
          'TAKE_PROFIT_LIMIT'],
         'icebergAllowed': True,
         'ocoAllowed': True,
         'quoteOrderQtyMarketAllowed': True,
         'allowTrailingStop': True,
         'cancelReplaceAllowed': True,
         'isSpotTradingAllowed': True,
         'isMarginTradingAllowed': True,
         'filters': [{'filterType': 'PRICE_FILTER',
           'minPrice': '0.01000000',
           'maxPrice': '1000000.00000000',
           'tickSize': '0.01000000'},
          {'filterType': 'PERCENT_PRICE',
           'multiplierUp': '5',
           'multiplierDown': '0.2',
           'avgPriceMins': 5},
          {'filterType': 'LOT_SIZE',
           'minQty': '0.00010000',
           'maxQty': '9000.00000000',
           'stepSize': '0.00010000'},
          {'filterType': 'MIN_NOTIONAL',
           'minNotional': '10.00000000',
           'applyToMarket': True,
           'avgPriceMins': 5},
          {'filterType': 'ICEBERG_PARTS', 'limit': 10},
          {'filterType': 'MARKET_LOT_SIZE',
           'minQty': '0.00000000',
           'maxQty': '6175.01628506',
           'stepSize': '0.00000000'},
          {'filterType': 'TRAILING_DELTA',
           'minTrailingAboveDelta': 10,
           'maxTrailingAboveDelta': 2000,
           'minTrailingBelowDelta': 10,
           'maxTrailingBelowDelta': 2000},
          {'filterType': 'MAX_NUM_ORDERS', 'maxNumOrders': 200},
          {'filterType': 'MAX_NUM_ALGO_ORDERS', 'maxNumAlgoOrders': 5}],
         'permissions': ['SPOT', 'MARGIN', 'TRD_GRP_004']}
    """
    return {k['symbol']: k for k in get_exchange_info()['symbols']}


###########
# account #
###########


def get_account_status(decimal_mode: bool,
                       api_key: str,
                       api_secret: str) -> dict:
    """
    Fetch account status detail.

    For Machine Learning limits, restrictions will be applied to account. If a user has been restricted by the ML system, they may
    check the reason and the duration by using the [/sapi/v1/account/status] endpoint

    :return dict: Example: `{ "data": "Normal" }`

    """
    endpoint = '/sapi/v1/account/status'
    return api_raw_signed_get(endpoint=endpoint,
                              weight=1,
                              decimal_mode=decimal_mode,
                              api_key=api_key,
                              api_secret=api_secret)


def get_margin_bnb_interest_status(decimal_mode: bool,
                                   api_key: str,
                                   api_secret: str) -> dict:
    """
    Get BNB Burn Status (USER_DATA)

    GET /sapi/v1/bnbBurn (HMAC SHA256)

    Weight(IP): 1

    :return dict: Example

    .. code-block::

           {
               "spotBNBBurn":true,
               "interestBNBBurn": false
            }

    """
    endpoint = '/sapi/v1/bnbBurn'
    return api_raw_signed_get(endpoint=endpoint,
                              weight=1,
                              decimal_mode=decimal_mode,
                              api_key=api_key,
                              api_secret=api_secret)


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

       .. code-block::

            {'x-mbx-used-weight-1m': 1200,
             'X-SAPI-USED-IP-WEIGHT-1M': 1200,
             'X-SAPI-USED-UID-WEIGHT-1M': 1200,
             'x-mbx-order-count-10s': 50,
             'x-mbx-order-count-1d': 160000,
             'x-mbx-used-weight': 6100}

    :return dict:
    """

    if not info_dict:
        info_dict = get_exchange_info()

    limits = info_dict['rateLimits']

    limits_dict = {}

    for i in limits:
        k3 = None
        if i['rateLimitType'].upper() == 'ORDERS':
            k1 = f"x-mbx-order-count-{i['intervalNum']}{i['interval'][0].lower()}"
            k2 = f"x-mbx-order-count-{i['intervalNum']}{i['interval'][0].lower()}"

        elif i['rateLimitType'].upper() == 'REQUEST_WEIGHT':
            k1 = f"x-mbx-used-weight-{i['intervalNum']}{i['interval'][0].lower()}"
            k2 = f"X-SAPI-USED-IP-WEIGHT-{i['intervalNum']}{i['interval'][0].upper()}"
            k3 = f"X-SAPI-USED-UID-WEIGHT-{i['intervalNum']}{i['interval'][0].upper()}"

        elif i['rateLimitType'].upper() == 'RAW_REQUESTS':
            k1 = "x-mbx-used-weight"
            k2 = "x-mbx-used-weight"
        else:
            raise Exception("BinPan Rate Limit not parsed")

        v = i['limit']

        limits_dict[k1] = v
        limits_dict[k2] = v
        if k3:
            limits_dict[k3] = v

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
    """
    Example:

    .. code-block:: Python

       exchange.get_filters('TLMBUSD', info_dic=info_dic)

       {'TLMBUSD': {'PRICE_FILTER_minPrice': '0.00001000',
          'PRICE_FILTER_maxPrice': '1000.00000000',
          'PRICE_FILTER_tickSize': '0.00001000',
          'PERCENT_PRICE_multiplierUp': '5',
          'PERCENT_PRICE_multiplierDown': '0.2',
          'PERCENT_PRICE_avgPriceMins': 5,
          'LOT_SIZE_minQty': '1.00000000',
          'LOT_SIZE_maxQty': '900000.00000000',
          'LOT_SIZE_stepSize': '1.00000000',
          'MIN_NOTIONAL_minNotional': '10.00000000',
          'MIN_NOTIONAL_applyToMarket': True,
          'MIN_NOTIONAL_avgPriceMins': 5,
          'ICEBERG_PARTS_limit': 10,
          'MARKET_LOT_SIZE_minQty': '0.00000000',
          'MARKET_LOT_SIZE_maxQty': '1181959.38584316',
          'MARKET_LOT_SIZE_stepSize': '0.00000000',
          'TRAILING_DELTA_minTrailingAboveDelta': 10,
          'TRAILING_DELTA_maxTrailingAboveDelta': 2000,
          'TRAILING_DELTA_minTrailingBelowDelta': 10,
          'TRAILING_DELTA_maxTrailingBelowDelta': 2000,
          'MAX_NUM_ORDERS_maxNumOrders': 200,
          'MAX_NUM_ALGO_ORDERS_maxNumAlgoOrders': 5}}

    :param dict info_dic: A BinPan exchange info dictionary

    :return dict: A dict with all flatten values.
    """
    if not info_dic:
        info_dic = get_info_dic()
    return {k: flatten_filter(v['filters']) for k, v in info_dic.items()}


def get_filters(symbol: str,
                info_dic: dict = None) -> dict:
    """
    For a symbol, get exchange filter conditions.

    :param str symbol: A Binance symbol.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call. It's optional to avoid an API call.
    :return a dict: A dictionary with exclusively data for a symbol.

    Example:

    .. code-block::

        from handlers import exchange

        filters = exchange.get_filters('ETHBTC')

        filters

         {'ETHBTC': {'PRICE_FILTER_minPrice': '0.00000100',
          'PRICE_FILTER_maxPrice': '922327.00000000',
          'PRICE_FILTER_tickSize': '0.00000100',
          'PERCENT_PRICE_multiplierUp': '5',
          'PERCENT_PRICE_multiplierDown': '0.2',
          'PERCENT_PRICE_avgPriceMins': 5,
          'LOT_SIZE_minQty': '0.00010000',
          'LOT_SIZE_maxQty': '100000.00000000',
          'LOT_SIZE_stepSize': '0.00010000',
          'MIN_NOTIONAL_minNotional': '0.00010000',
          'MIN_NOTIONAL_applyToMarket': True,
          'MIN_NOTIONAL_avgPriceMins': 5,
          'ICEBERG_PARTS_limit': 10,
          'MARKET_LOT_SIZE_minQty': '0.00000000',
          'MARKET_LOT_SIZE_maxQty': '1135.26522780',
          'MARKET_LOT_SIZE_stepSize': '0.00000000',
          'TRAILING_DELTA_minTrailingAboveDelta': 10,
          'TRAILING_DELTA_maxTrailingAboveDelta': 2000,
          'TRAILING_DELTA_minTrailingBelowDelta': 10,
          'TRAILING_DELTA_maxTrailingBelowDelta': 2000,
          'MAX_NUM_ORDERS_maxNumOrders': 200,
          'MAX_NUM_ALGO_ORDERS_maxNumAlgoOrders': 5}}
    """
    if not info_dic:
        info_dic = get_info_dic()
    return {k: flatten_filter(v['filters']) for k, v in info_dic.items() if k == symbol}


####################################
# symbol characteristics filtering #
####################################


def filter_tradeable(info_dic: dict) -> dict:
    """
    Returns, from BinPan exchange info dictionary currently trading symbols.

    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dict: BinPan exchange info dictionary, but, just with currently trading symbols.

    """
    return {k: v for k, v in info_dic.items() if v['status'].upper() == 'TRADING'}


def filter_spot(info_dic: dict) -> dict:
    """
    Returns, from BinPan exchange info dictionary currently SPOT symbols.

    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dict: BinPan exchange info dictionary, but, just with currently SPOT symbols.

    """
    return {k: v for k, v in info_dic.items() if 'SPOT' in v['permissions'] or 'spot' in v['permissions']}


def filter_margin(info_dic: dict) -> dict:
    """
    Returns, from BinPan exchange info dictionary currently MARGIN symbols.

    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dict: BinPan exchange info dictionary, but, just with currently MARGIN symbols.

    """
    return {k: v for k, v in info_dic.items() if 'MARGIN' in v['permissions'] or 'margin' in v['permissions']}


def filter_not_margin(symbols: list = None,
                      info_dic: dict = None) -> list:
    """
    Returns, from BinPan exchange info dictionary currently NOT MARGIN symbols.

    :param list symbols: A list of symbols to apply filter. Optional.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return list: BinPan exchange info dictionary, but, just with currently NOT MARGIN symbols.

    """
    if not info_dic:
        info_dic = get_info_dic()

    permissions_dic = {k: v['permissions'] for k, v in info_dic.items()}

    if symbols:
        return [s for s, p in permissions_dic.items() if 'MARGIN' in p and s in symbols]
    else:
        return [s for s, p in permissions_dic.items() if 'MARGIN' in p]


def filter_leveraged_tokens(info_dic: dict) -> dict:
    """
    Returns, from BinPan exchange info dictionary currently NOT LEVERAGED symbols.

    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dict: BinPan exchange info dictionary, but, just with currently NOT LEVERAGED symbols.

    """
    return {k: v for k, v in info_dic.items() if all(lev not in k for lev in ['UP', 'DOWN', 'BULL', 'BEAR'])}


def filter_legal(legal_coins: list,
                 info_dic: dict) -> dict:
    """
    Returns, from BinPan exchange info dictionary currently trading symbols not using legal Fiat money.

    :param list legal_coins: List of legal coins, Fiat coins. 
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dict: BinPan exchange info dictionary, but, without fiat legal symbols.

    """
    bases = get_bases_dic(info_dic=info_dic)
    quotes = get_quotes_dic(info_dic=info_dic)
    return {k: v for k, v in info_dic.items() if (not bases[k] in legal_coins) and (not quotes[k] in legal_coins)}


#################
# Exchange Data #
#################


def get_precision(info_dic: dict = None) -> dict:
    """
    Gets a dictionary with decimal positions each symbol.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dict: dictionary with decimal positions each symbol.
    """
    if not info_dic:
        info_dic = get_info_dic()
    return {k: {'baseAssetPrecision': v['baseAssetPrecision'],
                'quoteAssetPrecision': v['quoteAssetPrecision'],
                'baseCommissionPrecision': v['baseCommissionPrecision'],
                'quoteCommissionPrecision': v['quoteCommissionPrecision']}
            for k, v in info_dic.items()}


def get_orderTypes_and_permissions(info_dic: dict = None) -> dict:
    """
    Gets a dictionary with a list of order types suppoerted each symbol.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dict: dictionary with a list of order types suppoerted each symbol.
    """
    if not info_dic:
        info_dic = get_info_dic()
    return {k: {'orderTypes': v['orderTypes'], 'permissions': v['permissions']} for k, v in info_dic.items()}


def get_fees_dict(decimal_mode: bool,
                  api_key: str,
                  api_secret: str,
                  symbol: str = None
                  ) -> dict:
    """
    Returns fees for a symbol or for every symbol if not passed a symbol.

    :param str symbol: Optional to request just one symbol instead of all.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool decimal_mode: Fixes Decimal return type.
    :return dict: A dict with maker and taker fees.
    """
    endpoint = '/sapi/v1/asset/tradeFee'
    if symbol:
        symbol = symbol.upper()
    ret = api_raw_signed_get(endpoint,
                             params={'symbol': symbol},
                             weight=1,
                             decimal_mode=decimal_mode,
                             api_key=api_key,
                             api_secret=api_secret)
    if decimal_mode:
        return {i['symbol']: {'makerCommission': dd(i['makerCommission']),
                              'takerCommission': dd(i['takerCommission'])} for i in ret}
    else:
        return {i['symbol']: {'makerCommission': float(i['makerCommission']),
                              'takerCommission': float(i['takerCommission'])} for i in ret}


def get_fees(decimal_mode: bool,
             api_key: str,
             api_secret: str,
             symbol: str = None) -> pd.DataFrame:
    """
    Returns fees for a symbol or for every symbol if not passed.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param str symbol: Optional to request just one symbol instead of all.
    :return pd.DataFrame: A pandas dataframe with all the fees applied each symbol.
    """
    ret = get_fees_dict(symbol=symbol, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    return pd.DataFrame(ret).transpose()


def get_system_status():
    """
    Fetch system status.

    Weight(IP): 1

    :return dict: As shown in example.

    Example:

    .. code-block::

        {
           "status": 0,              // 0: normal，1：system maintenance
            "msg": "normal"           // "normal", "system_maintenance"
        }

    """
    return api_raw_get(endpoint='/sapi/v1/system/status',
                       weight=1)['msg']


def get_coins_and_networks_info(decimal_mode: bool,
                                api_key: str,
                                api_secret: str) -> tuple:
    """
    Get information of coins (available for deposit and withdraw) for user.

    GET /sapi/v1/capital/config/getall (HMAC SHA256)

    Weight(IP): 10

    :return pd.DataFrame, pd.DataFrame:
    """

    ret = api_raw_signed_get(endpoint='/sapi/v1/capital/config/getall',
                             weight=10, decimal_mode=decimal_mode,
                             api_key=api_key,
                             api_secret=api_secret)
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


def get_coins_info_list(decimal_mode: bool,
                        api_key: str,
                        api_secret: str,
                        coin: str = None) -> list:
    """
    Bring all coins exchange info in a list if no one is specified.

    Returns a list of dictionaries, one for each currency.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param str coin: Limit response to a coin.
    :return list: A list of dictionaries each coin.
    """

    endpoint = '/sapi/v1/capital/config/getall'
    check_weight(weight=10, endpoint=endpoint)
    ret = api_raw_signed_get(endpoint=endpoint,
                             decimal_mode=decimal_mode,
                             api_key=api_key,
                             api_secret=api_secret)
    if not coin:
        return ret
    else:
        return [c for c in ret if c['coin'].upper() == coin.upper()]


def get_coins_info_dic(decimal_mode: bool,
                       api_key: str,
                       api_secret: str,
                       coin: str = None) -> dict:
    """
    Useful managing coins info in a big dictionary with coins as keys.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param str coin: Limit response to a coin.
    :return list: A dictionary with each coin data as value.
    """
    coins_data_list = get_coins_info_list(coin=coin,
                                          decimal_mode=decimal_mode,
                                          api_key=api_key,
                                          api_secret=api_secret)
    return {c['coin']: c for c in coins_data_list}


def get_legal_coins(decimal_mode: bool,
                    api_key: str,
                    api_secret: str,
                    coins_dic: dict = None) -> list:
    """
    Fetch coins containing isLegalMoney=true

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict coins_dic: Avoid fetching the API by passing a dict with coins data.
    :return list: A list with coins names.
    """
    if not coins_dic:
        coins_dic = get_coins_info_dic(decimal_mode=decimal_mode,
                                       api_key=api_key,
                                       api_secret=api_secret)
    return [coin for coin, data in coins_dic.items() if data['isLegalMoney'] is True]


def get_leveraged_coins(decimal_mode: bool,
                        api_key: str,
                        api_secret: str,
                        coins_dic: dict = None) -> list:
    """
    Search for Binance leveraged coins by searching UP or DOWN before an existing coin, examples:

        .. code-block:: python

            ['1INCHDOWN', '1INCHUP', 'AAVEDOWN', 'AAVEUP', 'ADADOWN', ... ]

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict coins_dic: Avoid fetching the API by passing a dict with coins data.
    :return list: A list with leveraged coins names.
    """
    if not coins_dic:
        coins_dic = get_coins_info_dic(decimal_mode=decimal_mode,
                                       api_key=api_key,
                                       api_secret=api_secret)

    leveraged = []

    coins_up = [i + 'UP' for i in coins_dic.keys()]
    coins_down = [i + 'DOWN' for i in coins_dic.keys()]

    for coin, _ in coins_dic.items():
        if coin in coins_up:
            leveraged.append(coin)
        elif coin in coins_down:
            leveraged.append(coin)
    return leveraged


def get_leveraged_symbols(decimal_mode: bool,
                          api_key: str,
                          api_secret: str,
                          info_dic: dict = None, leveraged_coins: list = None) -> list:
    """
    Search for Binance symbols based on leveraged coins by searching UP or DOWN before an existing coin in symbol,
    leveraged coins examples are:

        .. code-block:: python

            # leveraged coins
            ['1INCHDOWN', '1INCHUP', 'AAVEDOWN', 'AAVEUP', 'ADADOWN', ... ]

            # leveraged symbols

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict info_dic: Avoid fetching the API by passing a dict with symbols data.
    :param list leveraged_coins: Avoid fetching the API for getting coins by passing a list with coins data.
    :return list: A list with leveraged coins names.
    """
    if not info_dic:
        info_dic = get_info_dic()
    if not leveraged_coins:
        leveraged_coins = get_leveraged_coins(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)

    bases = get_bases_dic(info_dic=info_dic)
    quotes = get_quotes_dic(info_dic=info_dic)
    leveraged_symbols = []
    for symbol in info_dic.keys():
        b = bases[symbol]
        q = quotes[symbol]
        if b in leveraged_coins or q in leveraged_coins:
            leveraged_symbols.append(symbol)
    return leveraged_symbols


#######################
# Exchange Statistics #
#######################


def get_quotes_dic(info_dic: dict = None) -> dict:
    """
    Get quote coin each symbol.

    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dictionary: It gets symbols as keys and quote coin as values.
    """
    if not info_dic:
        info_dic = get_info_dic()
    return {k: v['quoteAsset'] for k, v in info_dic.items()}


def get_bases_dic(info_dic: dict = None) -> dict:
    """
    Get base coin each symbol.

    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return dictionary: It gets symbols as keys and base coin as values.
    :param info_dic:
    :return:
    """
    if not info_dic:
        info_dic = get_info_dic()
    return {k: v['baseAsset'] for k, v in info_dic.items()}


def exchange_status(decimal_mode: bool,
                    api_key: str,
                    api_secret: str,
                    tradeable=True,
                    spot_required=True,
                    margin_required=True,
                    drop_legal=True,
                    filter_leveraged=True,
                    info_dic: dict = None,
                    symbol_filters: dict = None) -> tuple:
    """
    It returns a lot of results: bases_dic, quotes_dic, legal_coins, not_legal_pairs, symbol_filters, filtered_pairs

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool tradeable: Require or not just currently trading symbols.
    :param bool spot_required: Requires just SPOT currently trading symbols.
    :param bool margin_required: Requires just MARGIN currently trading symbols.
    :param bool drop_legal: Drops symbols with legal coins.
    :param bool filter_leveraged: Drops symbols with leveraged coins.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :param dict symbol_filters: A BinPan symbols filters dict.
    :return tuple: bases_dic, quotes_dic, legal_coins, not_legal_pairs, symbol_filters, filtered_pairs
    """
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

    legal_coins = get_legal_coins(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    not_legal_pairs = filter_legal(legal_coins=legal_coins, info_dic=info_dic)
    if drop_legal:
        filtered_info = filter_legal(legal_coins=legal_coins, info_dic=filtered_info)

    if not symbol_filters:
        symbol_filters = get_symbols_filters(info_dic=info_dic)

    filtered_pairs = filtered_info.keys()
    return bases_dic, quotes_dic, legal_coins, not_legal_pairs, symbol_filters, filtered_pairs


def get_24h_statistics(symbol: str = None) -> dict:  # 24h rolling window
    """
    GET /api/v3/ticker/24hr

    24 hour rolling window price change statistics. Careful when accessing this with no symbol.

    Weight(IP):

    Symbols requested weight:


    1-20:	        1

    21-100:	    20

    101 or more:	40

    If symbols parameter is omitted	40

    :param str symbol: Optional symbol.
    :return dict: As example shown.

    Example:

    .. code-block::

       {
          "symbol": "BNBBTC",
          "priceChange": "-94.99999800",
          "priceChangePercent": "-95.960",
          "weightedAvgPrice": "0.29628482",
          "prevClosePrice": "0.10002000",
          "lastPrice": "4.00000200",
          "lastQty": "200.00000000",
          "bidPrice": "4.00000000",
          "bidQty": "100.00000000",
          "askPrice": "4.00000200",
          "askQty": "100.00000000",
          "openPrice": "99.00000000",
          "highPrice": "100.00000000",
          "lowPrice": "0.10000000",
          "volume": "8913.30000000",
          "quoteVolume": "15.30000000",
          "openTime": 1499783499040,
          "closeTime": 1499869899040,
          "firstId": 28385,   // First tradeId
          "lastId": 28460,    // Last tradeId
          "count": 76         // Trade count
        }
    """
    endpoint = '/api/v3/ticker/24hr?'
    if symbol:
        check_weight(endpoint=endpoint, weight=1)
        return api_raw_get(endpoint=endpoint, params={'symbol': symbol}, weight=1)
    else:
        check_weight(endpoint=endpoint, weight=40)
        return api_raw_get(endpoint=endpoint, weight=40)


def not_iterative_coin_conversion(coin: str,
                                  decimal_mode: bool,
                                  prices: dict = None,
                                  try_coin: str = 'BTC',
                                  coin_qty: float = 1) -> float or None:
    """
    Converts any coin quantity value to a reference coin.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str coin: A coin to convert to other coin value.
    :param dict prices: Current prices of all symbols.
    :param str try_coin: Reference coin to convert value to.
    :param float coin_qty: A quantity to use in calculation.
    :return float: Converted value result.
    """
    if not prices:
        prices = get_prices_dic(decimal_mode=decimal_mode)
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


def convert_to_other_coin(coin: str,
                          decimal_mode: bool,
                          convert_to: str = 'USDT',
                          coin_qty: float = 1,
                          prices: dict = None) -> float:
    """
    Convert value of a quantity of coins to value in other coin.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str coin: Your coin.
    :param str convert_to: A Binance existing coin.
    :param float coin_qty: A quantity.
    :param dict prices: A dict with all current prices.
    :return float: Value expressed in the converted coin.
    """
    coin = coin.upper()
    convert_to = convert_to.upper()

    if coin == convert_to:
        return coin_qty

    if not prices:
        prices = get_prices_dic(decimal_mode=decimal_mode)

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
        ret1 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='BTC', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret1:
            return ret1
        ret2 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='BUSD', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret2:
            return ret2
        ret3 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='BNB', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret3:
            return ret3
        ret4 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='ETH', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret4:
            return ret4
        ret5 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='TUSD', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret5:
            return ret5
        ret6 = not_iterative_coin_conversion(coin=coin, prices=prices, try_coin='USDC', coin_qty=coin_qty, decimal_mode=decimal_mode)
        if ret6:
            return ret6
        else:
            return np.nan


def convert_symbol_base_to_other_coin(decimal_mode: bool,
                                      symbol_to_convert_base: str = 'ETHBTC',
                                      base_qty: float = 1,
                                      convert_to: str = 'USDT',
                                      prices: dict = None,
                                      info_dic: dict = None) -> float or dd:
    """
        Convert value of a quantity of coins to value in other coin.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str symbol_to_convert_base: A symbol to get it's base to convert to other coin value.
    :param float base_qty: A quantity.
    :param str convert_to: A Binance existing coin.
    :param dict prices: A dict with all current prices.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :return float: Value expressed in the converted coin.
    """
    if decimal_mode:
        my_type = dd
    else:
        my_type = float

    if not info_dic:
        info_dic = get_info_dic()

    # bases_dic = {k: v['baseAsset'] for k, v in info_dic.items()}
    bases_dic = get_bases_dic(info_dic=info_dic)
    try:
        base = bases_dic[symbol_to_convert_base]
    except KeyError:
        return 0

    if not prices:
        prices = get_prices_dic(decimal_mode=decimal_mode)
    ret = convert_to_other_coin(coin=base, convert_to=convert_to, coin_qty=base_qty, prices=prices, decimal_mode=decimal_mode)
    if type(ret) == int or type(ret) == float or type(ret) == dd:
        return my_type(ret)
    else:
        raise Exception('BinPan Error: convert_symbol_base_to_other_coin breakpoint')


def convert_utc_milliseconds(ms) -> str:
    seconds = int(ms) / 1000
    return str(datetime.utcfromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S.%f'))


def statistics_24h(decimal_mode: bool,
                   api_key: str,
                   api_secret: str,
                   tradeable=True,
                   spot_required=True,
                   margin_required=False,
                   drop_legal=True,
                   filter_leveraged=True,
                   info_dic=None,
                   stablecoin_value='BUSD',
                   sort_by: str = 'priceChangePercent') -> pd.DataFrame:
    """
    Generates a dataframe with the filters to apply with the statistics of the last 24 hours. Optionally, you can generate the column to
    convert the volume to USDT.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool tradeable: Require or not just currently trading symbols.
    :param bool spot_required: Requires just SPOT currently trading symbols.
    :param bool margin_required: Requires just MARGIN currently trading symbols.
    :param bool drop_legal: Drops symbols with legal coins.
    :param bool filter_leveraged: Drops symbols with leveraged coins.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :param stablecoin_value: StableCoin reference for value.
    :param str sort_by: A column to sort by. Default is 'priceChangePercent'.
    :return pd.DataFrame: Picture result.

    .. image:: images/exchange_statistics_24h.png
       :width: 1000
       :alt: exchange_statistics_24h.png

    """

    if not info_dic:
        info_dic = get_info_dic()

    bases_dic, quotes_dic, _, not_legal_pairs, _, filtered_pairs = exchange_status(tradeable=tradeable,
                                                                                   spot_required=spot_required,
                                                                                   margin_required=margin_required,
                                                                                   drop_legal=drop_legal,
                                                                                   filter_leveraged=filter_leveraged,
                                                                                   info_dic=info_dic,
                                                                                   decimal_mode=decimal_mode,
                                                                                   api_key=api_key,
                                                                                   api_secret=api_secret)
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
        prices = get_prices_dic(decimal_mode=decimal_mode)
        # info_dic = {k['symbol']: k for k in market.get_exchange_info()['symbols']}
        # bases_dic = market.get_bases_dic(info_dic=info_dic)
        # quotes_dic = market.get_quotes_dic(info_dic=info_dic)

        stable_coin_value_name = f"{stablecoin_value}_value"

        for pair in prices.keys():
            usdt_val = convert_symbol_base_to_other_coin(symbol_to_convert_base=pair,
                                                         base_qty=1,
                                                         convert_to=stablecoin_value,
                                                         prices=prices,
                                                         info_dic=info_dic,
                                                         decimal_mode=decimal_mode)

            df.loc[df['symbol'] == pair, stable_coin_value_name] = float(usdt_val)  # float for pandas

        stable_coin_volumen_name = f"{stablecoin_value}_volume"
        df[stable_coin_volumen_name] = df[stable_coin_value_name] * df['volume']

        return df.sort_values(stable_coin_volumen_name, ascending=False)

    return df.sort_values(sort_by, ascending=False)


def get_top_gainers(decimal_mode: bool,
                    api_key: str,
                    api_secret: str,
                    info_dic: dict = None,
                    tradeable=True,
                    spot_required=True,
                    margin_required=False,
                    drop_legal=True,
                    filter_leveraged=True,
                    top_gainers_qty: int = None,
                    my_quote: str = 'BUSD',
                    drop_stable_pairs: bool = True,
                    sort_by_column: str = 'priceChangePercent',
                    full_return: bool = False
                    ) -> pd.DataFrame:
    """
    Generates a dataframe for symbols against a quote with the filters to apply with the statistics of the last 24 hours.
    Optionally, you can generate the column to convert the volume to USDT.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool tradeable: Require or not just currently trading symbols.
    :param bool spot_required: Requires just SPOT currently trading symbols.
    :param bool margin_required: Requires just MARGIN currently trading symbols.
    :param bool drop_legal: Drops symbols with legal coins.
    :param bool filter_leveraged: Drops symbols with leveraged coins.
    :param dict info_dic: BinPan exchange info dictionary. It's optional to avoid an API call.
    :param top_gainers_qty: Limit result top lines.
    :param str my_quote: A quote coin to reference values.
    :param bool drop_stable_pairs: It drop stablecoin to stablecoin symbols.
    :param str sort_by_column: A column to sort by. Default is 'priceChangePercent'.
    :param full_return: Activate return full columns data from exchange, else, returns just a few basic columns.
    :return pd.DataFrame: A dataframe as in the example .

    Example:

    .. code-block::

        priceChangePercent	volume
        symbol
        SSVBUSD	    62.693	1.860517e+06
        SANTOSBUSD	28.914	2.218403e+06
        LDOBUSD	    19.774	5.933290e+06
        EOSBUSD	    17.525	6.134061e+06
        LAZIOBUSD	14.952	9.952292e+05
        ...	...	...
        WINGBUSD	-6.363	6.913284e+05
        NKNBUSD	    -7.632	7.375257e+07
        BONDBUSD	-7.872	9.194983e+05
        BTCSTBUSD	-15.242	5.525540e+05
        STGBUSD	    -15.663	1.879494e+07
        300 rows × 2 columns


    """
    if not info_dic:
        info_dic = get_info_dic()

    top_gainers = statistics_24h(decimal_mode=decimal_mode,
                                 api_key=api_key,
                                 api_secret=api_secret,
                                 tradeable=tradeable,
                                 spot_required=spot_required,
                                 margin_required=margin_required,
                                 drop_legal=drop_legal,
                                 info_dic=info_dic,
                                 filter_leveraged=filter_leveraged,
                                 stablecoin_value=my_quote)
    # # filter quote
    # top_gainers = top_gainers.loc[top_gainers['quote'] == my_quote]

    if drop_stable_pairs:
        stable_pairs = [f"{s}{my_quote}" for s in stablecoins]
        stable_pairs += [f"{my_quote}{s}" for s in stablecoins]

        top_gainers = top_gainers.loc[~top_gainers['symbol'].isin(stable_pairs)]

    if not full_return:
        top_gainers = top_gainers[['priceChangePercent', 'volume', f'{my_quote}_value', f'{my_quote}_volume']]

    if top_gainers_qty:
        return top_gainers.sort_values(sort_by_column, ascending=False).head(top_gainers_qty)
    else:
        return top_gainers.sort_values(sort_by_column, ascending=False)
