from .quest import api_raw_get, api_raw_signed_get
from .logs import Logs

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
