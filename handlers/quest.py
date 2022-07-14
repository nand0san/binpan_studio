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

try:
    global api_secret, api_key
    # noinspection PyUnresolvedReferences
    from secret import api_key, api_secret

except ImportError:
    print("""
No API Key or API Secret

API key would be needed for personal API calls. Any other calls will work.

Adding:

binpan.handlers.files_filters.add_api_key("xxxx")
binpan.handlers.files_filters.add_api_secret("xxxx")

API keys will be added to a file called secret.py in an encrypted way. API keys in memory stay encrypted except in the API call instant.
""")

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

current_limit_headers = {'X-SAPI-USED-IP-WEIGHT-1M': 0,
                         'x-mbx-used-weight': 0,
                         'x-mbx-used-weight-1m': 0}

aplicable_limits = {'X-SAPI-USED-IP-WEIGHT-1M': api_rate_limits['REQUEST_1M'],
                    'x-mbx-used-weight': api_rate_limits['REQUEST_5M'],
                    'x-mbx-used-weight-1m': api_rate_limits['REQUEST_1M']}

api_limits_weight_decrease_per_seconds = {'X-SAPI-USED-IP-WEIGHT-1M': api_rate_limits['REQUEST_1M'] // 60,
                                          'x-mbx-used-weight-1m': api_rate_limits['REQUEST_1M'] // 60,
                                          'x-mbx-used-weight': api_rate_limits['REQUEST_5M'] // (60 * 5)}  # is the five minutes api limit?


# TODO: identify headers for order endpoints

#################
# Aux functions #
#################


def update_weights(headers, father_url: str):
    global current_limit_headers

    weight_logger.debug(f"Header: {headers}")
    weight_headers = {k: int(v) for k, v in headers.items() if 'WEIGHT' in k.upper() or 'COUNT' in k.upper()}
    weight_headers = {k: 0 if k not in weight_headers.keys() else v for k, v in current_limit_headers.items()}

    # register for detecting new weight headers
    weight_logger.info(f"Father url: {father_url} Weights updated from API: {weight_headers}")

    for k, v in weight_headers.items():
        current_limit_headers[k] = v
    weight_logger.info(f"Current weights registered: {current_limit_headers}")


def check_weight(weight: int,
                 endpoint: str,
                 no_wait=False):
    global api_rate_limits, current_limit_headers

    weight_logger.debug(f"Checking weight for {endpoint}")

    test_headers = copy.deepcopy(current_limit_headers)

    # identify
    if '/api/v3/order/' in endpoint:
        header_to_update = ''  # TODO: unknown yet
        test_headers[header_to_update] += weight
        waitable_for = [header_to_update]
    elif '/sapi/v1/' in endpoint:
        header_to_update = 'X-SAPI-USED-IP-WEIGHT-1M'
        test_headers[header_to_update] += weight
        waitable_for = [header_to_update]
    elif '/api/v3/' in endpoint:
        header_to_update = 'x-mbx-used-weight-1m'
        test_headers[header_to_update] += weight
        waitable_for = [header_to_update]
        header_to_update = 'x-mbx-used-weight'
        test_headers[header_to_update] += weight
        waitable_for.append(header_to_update)
    else:
        msg = f"BinPan error: Unknown endpoint for weight check: {endpoint}"
        weight_logger.error(msg)
        raise Exception(msg)

    # check & wait
    weight_logger.debug(f"BEFORE WAIT Current weights registered: {current_limit_headers}")

    for head, value in test_headers.items():
        if value > aplicable_limits[head] and current_limit_headers[head] > 0:
            excess = ((value - aplicable_limits[head]) // api_limits_weight_decrease_per_seconds[head]) + 2
            if not no_wait and head in waitable_for:
                weight_logger.warning(f"{head}={value} > {aplicable_limits[head]} Current value: {current_limit_headers[head]}")
                weight_logger.warning(f"Waiting {excess} seconds for {head} to decrease rate.")
                sleep(excess)
            current_limit_headers[head] = aplicable_limits[head]
        else:
            current_limit_headers[head] += weight  # should be updated with response headers

    weight_logger.debug(f"Checked and waited:  {current_limit_headers}")


def get_server_time(no_wait=False):
    endpoint = '/api/v3/time'
    check_weight(1, endpoint=endpoint, no_wait=no_wait)
    return int(get_response(url=endpoint)['serverTime'])


##################
# ## Requests ## #
##################


def convert_response_type(response_data: dict or list) -> dict or list:
    """Cambia el tipo de respuestas de la api"""
    if type(response_data) == dict:
        response = response_data.copy()
        for k, v in response.items():
            if k in float_api_items:
                response[k] = float(v)
            elif k in int_api_items:
                response[k] = int(v)
            elif type(v) == dict:
                response[k] = convert_response_type(v)
            elif type(v) == list:
                response[k] = [convert_response_type(ii) for ii in v]
    elif type(response_data) == list:
        response = [convert_response_type(iii) for iii in response_data]
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


def hashed_signature(url_params):  # los params para la signatura son los params de la timestamp
    return hmac.new(cipher_object.decrypt(api_secret).encode('utf-8'), url_params.encode('utf-8'),
                    hashlib.sha256).hexdigest()


# def float_to_string(num: float, d_max=20) -> str:
#     ctx = decimal.Context()
#     ctx.prec = d_max
#     num_str = ctx.create_decimal(repr(num))
#     return format(num_str, 'f')


def sign_request(params: dict, recvWindow: int) -> (list, dict):
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
    signature = hashed_signature(urlencode(params_tuples))
    params_tuples.append(("signature", signature,))  # y la añado como parámetro

    return params_tuples, headers


def get_signed_request(url: str,
                       params: dict or tuple = None,
                       recvWindow: int = 10000):
    """
    Hace un get firmado a una url junto con un diccionario de parámetros
    Para evitar errores de orden de parámetros se pasan en formato tupla
        https://dev.binance.vision/t/faq-signature-for-this-request-is-not-valid/176/4

    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow)
    ret = get_response(url, params=params_tuples, headers=headers)
    quest_logger.debug("get_signed_request: params_tuples: " + str(params_tuples))
    quest_logger.debug("get_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret)


def get_semi_signed_request(url, params=None):
    """Requests get with api key header and params"""
    headers = {"X-MBX-APIKEY": cipher_object.decrypt(api_key)}
    ret = get_response(url=url, params=params, headers=headers)
    quest_logger.debug("get_semi_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret)


def post_signed_request(url: str, params: dict = None, recvWindow: int = 10000):
    """
    Hace un POST firmado a una url junto con un diccionario de parámetros
    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow)
    ret = post_response(url, params=params_tuples, headers=headers)
    quest_logger.debug("post_signed_request: params_tuples: " + str(params_tuples))
    quest_logger.debug("get_semi_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret)


def delete_signed_request(url: str, params: dict = None, recvWindow: int = 10000):
    """
    Hace un DELETE firmado a una url junto con un diccionario de parámetros
    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow)
    ret = delete_response(url, params=params_tuples, headers=headers)
    quest_logger.debug("delete_signed_request: params_tuples: " + str(params_tuples))
    quest_logger.debug("get_semi_signed_request: headers: " + str(headers.keys()))
    return convert_response_type(ret)


# def check_tick_interval(tick_interval):
#     if not (tick_interval in tick_interval_values):
#         raise Exception(f"BinPan Error on tick_interval: {tick_interval} not in "
#                         f"expected API intervals.\n{tick_interval_values}")

####################
# Generic Requests #
####################

def api_raw_get(endpoint: str,
                weight: int,
                base_url: str = '',
                params: dict = None,
                headers: dict = None,

                no_wait=False):
    check_weight(weight, endpoint=base_url + endpoint, no_wait=no_wait)

    return get_response(url=base_url + endpoint,
                        params=params,
                        headers=headers)


def api_raw_signed_get(endpoint: str,
                       base_url: str = '',
                       params: dict = None,
                       weight: int = 1,
                       no_wait=False):
    check_weight(weight, endpoint=base_url + endpoint, no_wait=no_wait)
    return get_signed_request(url=base_url + endpoint,
                              params=params)


def api_raw_signed_post(endpoint: str,
                        base_url: str = '',
                        params: dict = None,
                        weight: int = 1,
                        no_wait=False):
    check_weight(weight, endpoint=base_url + endpoint, no_wait=no_wait)
    return post_signed_request(url=base_url + endpoint,
                               params=params)
