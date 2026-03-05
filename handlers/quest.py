# coding=utf-8
"""

Authenticated API requests module.

Public API requests are handled by panzer (BinancePublicClient).
This module only handles signed/semi-signed requests that require API keys.

"""
from urllib.parse import urlencode
import requests
import hmac
import hashlib

from .logs import LogManager
from .exceptions import BinanceAPIException, BinanceRequestException
from .starters import AesCipher

quest_logger = LogManager(filename='./logs/quest.log', name='quest', info_level='INFO')

base_url = 'https://api.binance.com'

# tick_seconds se mantiene aquí por compatibilidad con imports existentes (legacy)
tick_seconds = {'1m': 60, '3m': 60 * 3, '5m': 5 * 60, '15m': 15 * 60, '30m': 30 * 60, '1h': 60 * 60, '2h': 60 * 60 * 2,
                '4h': 60 * 60 * 4, '6h': 60 * 60 * 6, '8h': 60 * 60 * 8, '12h': 60 * 60 * 12, '1d': 60 * 60 * 24,
                '3d': 60 * 60 * 24 * 3, '1w': 60 * 60 * 24 * 7, '1M': 60 * 60 * 24 * 30}

float_api_items = ['price', 'origQty', 'executedQty', 'cummulativeQuoteQty', 'stopLimitPrice', 'stopPrice', 'commission', 'qty',
                   'origQuoteOrderQty', 'makerCommission', 'takerCommission']
int_api_items = ['orderId', 'orderListId', 'transactTime', 'tradeId', 'transactionTime', 'updateTime', 'time']

# cipher_object se inicializa lazy para evitar get_cpu_info() al importar
_cipher_object = None


def _get_cipher():
    global _cipher_object
    if _cipher_object is None:
        _cipher_object = AesCipher()
    return _cipher_object


class _CipherProxy:
    def decrypt(self, *args, **kwargs):
        return _get_cipher().decrypt(*args, **kwargs)

    def encrypt(self, *args, **kwargs):
        return _get_cipher().encrypt(*args, **kwargs)


cipher_object = _CipherProxy()


##################
# ## Requests ## #
##################


def convert_response_type(response_data: dict | list,
                          decimal_mode: bool) -> dict | list:
    """
    Infers types into api response and changes those.

    :param dict or list response_data: API response json loaded.
    :param bool decimal_mode: Sets numeric data type to decimal.
    :return dict or list: Typed API data in response.
    """
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


def handle_api_response(response) -> dict | list:
    """
    Raises if there is any problem with the server response or returns the raw json.

    :return: API response in dict or list format.
    """
    quest_logger.debug(response)
    if not (200 <= response.status_code < 300):
        quest_logger.error(response)
        raise BinanceAPIException(response, response.status_code, response.text)
    try:
        return response.json()
    except ValueError:
        quest_logger.error(response)
        raise BinanceRequestException(f'Invalid Response: {response.text}')


def _get_response(url: str, params: dict | list[tuple] = None, headers: dict = None) -> dict | list:
    """
    Internal GET request with response handling.
    """
    if not url.startswith(base_url):
        from urllib.parse import urljoin
        url = urljoin(base_url, url)
    response = requests.get(url, params=params, headers=headers)
    return handle_api_response(response)


def _post_response(url: str, params: dict | list[tuple] = None, headers: dict = None) -> dict | list:
    """
    Internal POST request with response handling.
    """
    if not url.startswith(base_url):
        from urllib.parse import urljoin
        url = urljoin(base_url, url)
    response = requests.post(url, params=params, headers=headers)
    return handle_api_response(response)


def _delete_response(url: str, params: dict | list[tuple] = None, headers: dict = None) -> dict | list:
    """
    Internal DELETE request with response handling.
    """
    if not url.startswith(base_url):
        from urllib.parse import urljoin
        url = urljoin(base_url, url)
    response = requests.delete(url, params=params, headers=headers)
    return handle_api_response(response)


############################
# Signature and Auth utils #
############################


def get_server_time() -> int:
    """
    Get time from server using panzer.

    :return int: A linux timestamp in milliseconds.
    """
    from .market import _get_panzer
    client = _get_panzer()
    return client.server_time()


def hashed_signature(url_params: str, api_secret: str) -> str:
    """
    Hashes params of a request with encoded API secret.

    :param str url_params: String of params to encode.
    :param str api_secret: Encoded API secret.
    :return str: Hashed signature.
    """
    return hmac.new(cipher_object.decrypt(api_secret).encode('utf-8'), url_params.encode('utf-8'),
                    hashlib.sha256).hexdigest()


def sign_request(params: dict | list[tuple],
                 recvWindow: int | None,
                 api_key: str,
                 api_secret: str
                 ) -> tuple[list, dict]:
    """
    Add signature to the request. Returns a list of params in tuples and a headers dict.

    :param dict or list params: Params for the request.
    :param int or None recvWindow: Milliseconds of life for the request to be responded.
    :param str api_key: Encoded API key.
    :param str api_secret: Encoded API secret.
    :return tuple: List of params in tuples and a headers dict.
    """
    quest_logger.debug(f"parse_request: {params}")

    if params is None:
        params = {}

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
    params_tuples.append(("timestamp", server_time_int,))
    signature = hashed_signature(urlencode(params_tuples), api_secret=api_secret)
    params_tuples.append(("signature", signature,))

    return params_tuples, headers


##############################
# Signed request functions   #
##############################


def get_signed_request(url: str,
                       decimal_mode: bool,
                       api_key: str,
                       api_secret: str,
                       params: dict | list[tuple] = None,
                       recvWindow: int = 10000
                       ) -> dict | list:
    """
    Signed GET request.

    :param str url: API endpoint.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param str api_key: Encoded API key.
    :param str api_secret: Encoded API secret.
    :param dict params: Params for the request.
    :param int recvWindow: Milliseconds of life for the request.
    :return dict or list: API response.
    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow, api_key=api_key, api_secret=api_secret)
    ret = _get_response(url, params=params_tuples, headers=headers)
    return convert_response_type(ret, decimal_mode=decimal_mode)


def get_semi_signed_request(url: str,
                            decimal_mode: bool,
                            api_key: str,
                            params: dict = None) -> dict | list:
    """
    GET with api key header (no full signature).

    :param str url: API endpoint.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param str api_key: Encoded API key.
    :param dict params: Params for the request.
    :return dict or list: API response.
    """
    headers = {"X-MBX-APIKEY": cipher_object.decrypt(api_key)}
    ret = _get_response(url=url, params=params, headers=headers)
    return convert_response_type(ret, decimal_mode=decimal_mode)


def post_signed_request(url: str,
                        decimal_mode: bool,
                        api_key: str,
                        api_secret: str,
                        params: dict = None,
                        recvWindow: int = 10000) -> dict | list:
    """
    Signed POST request.

    :param str url: API endpoint.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param str api_key: Encoded API key.
    :param str api_secret: Encoded API secret.
    :param dict params: Params for the request.
    :param int recvWindow: Milliseconds of life for the request.
    :return dict or list: API response.
    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow, api_key=api_key, api_secret=api_secret)
    ret = _post_response(url, params=params_tuples, headers=headers)
    return convert_response_type(ret, decimal_mode=decimal_mode)


def delete_signed_request(url: str,
                          decimal_mode: bool,
                          api_key: str,
                          api_secret: str,
                          params: dict = None,
                          recvWindow: int = 10000) -> dict | list:
    """
    Signed DELETE request.

    :param str url: API endpoint.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param str api_key: Encoded API key.
    :param str api_secret: Encoded API secret.
    :param dict params: Params for the request.
    :param int recvWindow: Milliseconds of life for the request.
    :return dict or list: API response.
    """
    params_tuples, headers = sign_request(params=params, recvWindow=recvWindow, api_key=api_key, api_secret=api_secret)
    ret = _delete_response(url, params=params_tuples, headers=headers)
    quest_logger.info("delete_signed_request: params_tuples: " + str(params_tuples))
    return convert_response_type(ret, decimal_mode=decimal_mode)


##################################
# Shortcuts for signed requests  #
##################################


def api_raw_signed_get(endpoint: str,
                       decimal_mode: bool,
                       api_key: str,
                       api_secret: str,
                       base_url: str = '',
                       params: dict | list[tuple] = None,
                       weight: int = 1) -> dict | list:
    """
    Shortcut to request signed GET to API.

    :param str endpoint: API endpoint to request.
    :param bool decimal_mode: If True, declares format of numbers to decimal type.
    :param str api_key: Encoded API key.
    :param str api_secret: Encoded API secret.
    :param str base_url: Base URL.
    :param dict params: Params for the request.
    :param int weight: Expected weight for the request.
    :return dict or list: API response.
    """
    return get_signed_request(url=base_url + endpoint,
                              params=params, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)


def api_raw_signed_post(endpoint: str,
                        decimal_mode: bool,
                        api_key: str,
                        api_secret: str,
                        base_url: str = '',
                        params: dict | list[tuple] = None,
                        weight: int = 1) -> dict | list:
    """
    Shortcut to request signed POST to API.

    :param str endpoint: API endpoint to request.
    :param bool decimal_mode: If True, declares format of numbers to decimal type.
    :param str api_key: Encoded API key.
    :param str api_secret: Encoded API secret.
    :param str base_url: Base URL.
    :param dict params: Params for the request.
    :param int weight: Expected weight for the request.
    :return dict or list: API response.
    """
    return post_signed_request(url=base_url + endpoint,
                               params=params, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
