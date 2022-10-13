# coding=utf-8
from .logs import Logs

from .exceptions import BinanceAPIException, BinanceRequestException

from .starters import AesCipher, get_exchange_limits

from urllib.parse import urljoin, urlencode
import requests
import hmac
import hashlib
import copy
from time import sleep
# from sys import path
# from os import getcwd
#
# path.append(getcwd())
#
# try:
#     global api_secret, api_key
#     # noinspection PyUnresolvedReferences
#     from secret import api_key, api_secret
#
# except ImportError:
#     print("""
# No API Key or API Secret
#
# API key would be needed for personal API calls. Any other calls will work.
#
# Adding:
#
# binpan.handlers.files.add_api_key("xxxx")
# binpan.handlers.files.add_api_secret("xxxx")
#
# API keys will be added to a file called secret.py in an encrypted way. API keys in memory stay encrypted except in the API call instant.
# """)

float_api_items = ['price', 'origQty', 'executedQty', 'cummulativeQuoteQty', 'stopLimitPrice', 'stopPrice', 'commission', 'qty',
                   'origQuoteOrderQty', 'makerCommission', 'takerCommission']
int_api_items = ['orderId', 'orderListId', 'transactTime', 'tradeId', 'transactionTime', 'updateTime', 'time']

base_url = 'https://api.binance.com'

tick_seconds = {'1m': 60, '3m': 60 * 3, '5m': 5 * 60, '15m': 15 * 60, '30m': 30 * 60, '1h': 60 * 60, '2h': 60 * 60 * 2,
                '4h': 60 * 60 * 4, '6h': 60 * 60 * 6, '8h': 60 * 60 * 8, '12h': 60 * 60 * 12, '1d': 60 * 60 * 24,
                '3d': 60 * 60 * 24 * 3, '1w': 60 * 60 * 24 * 7, '1M': 60 * 60 * 24 * 30}

tick_interval_values = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

weight_logger = Logs(filename='./logs/weight.log', name='weight', info_level='INFO')
quest_logger = Logs(filename='./logs/quest.log', name='quest', info_level='INFO')

cipher_object = AesCipher()

# rate limits
api_rate_limits = get_exchange_limits()

current_weight = {}
endpoint_headers = {}  # read temp file

aplicable_limits = {'X-SAPI-USED-IP-WEIGHT-1M': api_rate_limits['REQUEST_1M'],
                    'X-SAPI-USED-UID-WEIGHT-1M': api_rate_limits['REQUEST_1M'],
                    'x-mbx-used-weight': api_rate_limits['REQUEST_5M'],
                    'x-mbx-used-weight-1m': api_rate_limits['REQUEST_1M'],
                    'x-mbx-order-count-10s': api_rate_limits['ORDERS_10S'],
                    'x-mbx-order-count-1d': api_rate_limits['ORDERS_1D']}

api_limits_weight_decrease_per_seconds = {'X-SAPI-USED-IP-WEIGHT-1M': api_rate_limits['REQUEST_1M'] // 60,
                                          'X-SAPI-USED-UID-WEIGHT-1M': api_rate_limits['REQUEST_1M'] // 60,
                                          'x-mbx-used-weight-1m': api_rate_limits['REQUEST_1M'] // 60,
                                          'x-mbx-used-weight': api_rate_limits['REQUEST_5M'] // (60 * 5),
                                          'x-mbx-order-count-10s': api_rate_limits['ORDERS_10S'] // 10,
                                          'x-mbx-order-count-1d': api_rate_limits['ORDERS_1D'] // (
                                                      24 * 60 * 60)}  # is the five minutes api limit?


# TODO: identify headers for order endpoints

#################
# Aux functions #
#################


def add_header_for_endpoint(endpoint: str, header: str):
    global endpoint_headers

    if endpoint in endpoint_headers.keys():
        my_headers = endpoint_headers[endpoint]
        if header not in my_headers:
            my_headers.append(header)
            endpoint_headers.update({endpoint: my_headers})
    else:
        endpoint_headers.update({endpoint: [header]})
        weight_logger.debug(f"New endpoint headers control: {endpoint} -> {endpoint_headers[endpoint]}")


def update_weights(headers, father_url: str):
    global current_weight

    weight_logger.debug(f"Header: {headers}")

    # get weight headers from headers
    weight_headers = {k: int(v) for k, v in headers.items() if 'WEIGHT' in k.upper() or 'COUNT' in k.upper()}
    weight_logger.debug(f"Father url: {father_url} Weights updated from API: {weight_headers}")

    for head, value in weight_headers.items():
        add_header_for_endpoint(father_url, head)

    # decrease all other weights
    for k, v in current_weight.items():
        new_v = max(0, v - 1)  # is possible to decrease too much not used endpoints when using a variety of them
        current_weight.update({k: new_v})

    # annotate new weights updated
    for k, v in weight_headers.items():
        current_weight.update({k: v})

    weight_logger.debug(f"Current weights updated: {current_weight}")


def check_weight(weight: int,
                 endpoint: str):
    global current_weight, endpoint_headers, aplicable_limits, api_limits_weight_decrease_per_seconds

    weight_logger.debug(f"Checking weight for {endpoint}")

    future_weights = copy.deepcopy(current_weight)

    if not future_weights:  # cold start
        return
    weight_logger.debug(f"Future weight headers: {future_weights}")

    # añade los headers nuevos al archivo de endpoint headers
    for future_key in future_weights.keys():
        add_header_for_endpoint(endpoint, future_key)

    # busca los headers a incrementar usando el archivo de endpoint headers
    for endpoint_archived, end_headers_list in endpoint_headers.items():
        if endpoint_archived == endpoint:
            for head in end_headers_list:
                future_weights[head] += weight

    # check if needed header for this endpoint is overloaded
    aplicable_headers = endpoint_headers[endpoint]
    for aplicable_head in aplicable_headers:
        curr_limit = aplicable_limits[aplicable_head]  # must raise if new header in limits control
        future_weight = future_weights[aplicable_head]
        if future_weight >= curr_limit:
            excess = ((future_weight - curr_limit) // api_limits_weight_decrease_per_seconds[aplicable_head]) + 2

            weight_logger.warning(
                f"{aplicable_head}={future_weight} > {aplicable_limits[aplicable_head]} Current value: {current_weight[aplicable_head]}")
            weight_logger.warning(f"Waiting {excess} seconds for {aplicable_head} to decrease rate.")
            sleep(excess)


def get_server_time():
    endpoint = '/api/v3/time'
    check_weight(1, endpoint=endpoint)
    return int(get_response(url=endpoint)['serverTime'])


##################
# ## Requests ## #
##################


def convert_response_type(response_data: dict or list,
                          decimal_mode: bool) -> dict or list:
    """Cambia el tipo de respuestas de la api"""
    if decimal_mode:
        return response_data

    if type(response_data) == dict:
        response = response_data.copy()
        for k, v in response.items():
            if k in float_api_items:
                response[k] = float(v)
            elif k in int_api_items:
                response[k] = int(v)
            elif type(v) == dict:
                response[k] = convert_response_type(v, decimal_mode=decimal_mode)
            elif type(v) == list:
                response[k] = [convert_response_type(ii, decimal_mode=decimal_mode) for ii in v]
    elif type(response_data) == list:
        response = [convert_response_type(iii, decimal_mode=decimal_mode) for iii in response_data]
    else:
        response = response_data
    return response


def get_response(url: str,
                 params: dict or tuple = None,
                 headers: dict = None):
    if not url.startswith(base_url):
        url = urljoin(base_url, url)

    response = requests.get(url, params=params, headers=headers)
    update_weights(headers=response.headers,
                   father_url=url)
    quest_logger.debug(f"get_response parameters: {locals()}")

    return handle_api_response(response)


def post_response(url, params=None, headers=None):
    quest_logger.debug(f"post_response: params: {params} headers: {headers}")
    if not url.startswith(base_url):
        url = urljoin(base_url, url)
    response = requests.post(url, params=params, headers=headers)
    update_weights(response.headers, father_url=url)
    quest_logger.debug(f"post_response: response: {response}")
    return handle_api_response(response)


def delete_response(url, params: list = None, headers: dict = None):
    if not url.startswith(base_url):
        url = urljoin(base_url, url)
    response = requests.delete(url, params=params, headers=headers)
    update_weights(response.headers, father_url=url)
    quest_logger.debug(f"delete_response: response: {response}")
    return handle_api_response(response)


def handle_api_response(response) -> dict or list:
    """
    Raises if there is any problem with the server response or returns a json.

    :return: dict
    """
    if not (200 <= response.status_code < 300):
        quest_logger.error(response)
        raise BinanceAPIException(response, response.status_code, response.text)
    try:
        return response.json()
    except ValueError:
        quest_logger.error(response)
        raise BinanceRequestException(f'Invalid Response: {response.text}')


def hashed_signature(url_params: str,
                     api_secret: str):  # los params (no se el type bien) para la signatura son los params de la timestamp
    return hmac.new(cipher_object.decrypt(api_secret).encode('utf-8'), url_params.encode('utf-8'),
                    hashlib.sha256).hexdigest()


def sign_request(params: dict,
                 recvWindow: int,
                 api_key: str,
                 api_secret: str
                 ) -> (list, dict):
    quest_logger.debug(f"parse_request: {params}")

    if params is None:
        params = {}

    # clean none params
    params.update({'recvWindow': recvWindow})
    params = {k: v for k, v in params.items() if v is not None}

    params_tuples = []
    for k, v in params.items():
        if type(v) != list:
            params_tuples.append((k, v,))
        else:
            for i in v:
                params_tuples.append((k, i,))

    key = cipher_object.decrypt(api_key)
    headers = {"X-MBX-APIKEY": key}

    server_time_int = get_server_time()
    params_tuples.append(("timestamp", server_time_int,))  # añado el timestamp como parámetro
    signature = hashed_signature(urlencode(params_tuples), api_secret=api_secret)
    params_tuples.append(("signature", signature,))  # y la añado como parámetro

    return params_tuples, headers


def get_signed_request(url: str,
                       decimal_mode: bool,
                       api_key: str,
                       api_secret: str,
                       params: dict or tuple = None,
                       recvWindow: int = 10000):
    """
    Hace un get firmado a una url junto con un diccionario de parámetros
    Para evitar errores de orden de parámetros se pasan en formato tuple
        https://dev.binance.vision/t/faq-signature-for-this-request-is-not-valid/176/4

    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow, api_key=api_key, api_secret=api_secret)
    ret = get_response(url, params=params_tuples, headers=headers)
    quest_logger.debug("get_signed_request: params_tuples: " + str(params_tuples))
    quest_logger.debug("get_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret, decimal_mode=decimal_mode)


def get_semi_signed_request(url: str,
                            decimal_mode: bool,
                            api_key: str,
                            params: dict = None):
    """Requests get with api key header and params"""
    headers = {"X-MBX-APIKEY": cipher_object.decrypt(api_key)}
    ret = get_response(url=url, params=params, headers=headers)
    quest_logger.debug("get_semi_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret, decimal_mode=decimal_mode)


def post_signed_request(url: str,
                        decimal_mode: bool,
                        api_key: str,
                        api_secret: str,
                        params: dict = None,
                        recvWindow: int = 10000):
    """
    Hace un POST firmado a una url junto con un diccionario de parámetros
    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow, api_key=api_key, api_secret=api_secret)
    ret = post_response(url, params=params_tuples, headers=headers)
    quest_logger.debug("post_signed_request: params_tuples: " + str(params_tuples))
    quest_logger.debug("get_semi_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret, decimal_mode=decimal_mode)


def delete_signed_request(url: str,
                          decimal_mode: bool,
                          api_key: str,
                          api_secret: str,
                          params: dict = None,
                          recvWindow: int = 10000):
    """
    Hace un DELETE firmado a una url junto con un diccionario de parámetros
    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow, api_key=api_key, api_secret=api_secret)
    ret = delete_response(url, params=params_tuples, headers=headers)
    quest_logger.info("delete_signed_request: params_tuples: " + str(params_tuples))
    quest_logger.debug("get_semi_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret, decimal_mode=decimal_mode)


####################
# Generic Requests #
####################

def api_raw_get(endpoint: str,
                weight: int,
                base_url: str = '',
                params: dict = None,
                headers: dict = None):
    check_weight(weight, endpoint=base_url + endpoint)

    return get_response(url=base_url + endpoint,
                        params=params,
                        headers=headers)


def api_raw_signed_get(endpoint: str,
                       decimal_mode: bool,
                       api_key: str,
                       api_secret: str,
                       base_url: str = '',
                       params: dict = None,
                       weight: int = 1):
    check_weight(weight, endpoint=base_url + endpoint)
    return get_signed_request(url=base_url + endpoint,
                              params=params, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)


def api_raw_signed_post(endpoint: str,
                        decimal_mode: bool,
                        api_key: str,
                        api_secret: str,
                        base_url: str = '',
                        params: dict = None,
                        weight: int = 1):
    check_weight(weight, endpoint=base_url + endpoint)
    return post_signed_request(url=base_url + endpoint,
                               params=params, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
