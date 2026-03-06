# coding=utf-8
"""
Authenticated API requests module.

Public API requests are handled by panzer (BinancePublicClient).
Signed requests are handled by panzer (BinanceClient) since panzer 2.1.0.

This module provides thin wrappers that add BinPan-specific type conversion
(convert_response_type) on top of panzer's signed_request().

Credentials are managed by panzer's CredentialManager (~/.panzer_creds).
"""

from panzer import BinanceClient

from ..core.logs import LogManager

quest_logger = LogManager(filename='./logs/quest.log', name='quest', info_level='INFO')

float_api_items = ['price', 'origQty', 'executedQty', 'cummulativeQuoteQty', 'stopLimitPrice', 'stopPrice', 'commission', 'qty',
                   'origQuoteOrderQty', 'makerCommission', 'takerCommission']
int_api_items = ['orderId', 'orderListId', 'transactTime', 'tradeId', 'transactionTime', 'updateTime', 'time']


#############################
# Cliente autenticado panzer
#############################

_binance_client = None


def _get_binance_client(market: str = "spot") -> BinanceClient:
    """Returns the shared authenticated panzer client, creating it on first use."""
    global _binance_client
    if _binance_client is None:
        _binance_client = BinanceClient(market=market)
    return _binance_client


######################
# Conversión de tipos
######################


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


#####################
# Helpers internos
#####################


def _dict_to_tuples(params: dict | None) -> list[tuple[str, str | int]] | None:
    """
    Converts a dict of params to list of tuples for panzer's signed_request.
    Filters out None values and expands list values into multiple tuples.
    """
    if params is None:
        return None
    result = []
    for k, v in params.items():
        if v is None:
            continue
        if type(v) == list:
            for item in v:
                result.append((k, item))
        else:
            result.append((k, v))
    return result or None


##############################
# Wrappers de peticiones firmadas
##############################


def signed_get(endpoint: str,
               decimal_mode: bool,
               params: dict | None = None,
               recv_window: int = 10000) -> dict | list:
    """
    Signed GET request via panzer BinanceClient.

    :param str endpoint: API endpoint path.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param dict params: Params for the request.
    :param int recv_window: Milliseconds of life for the request.
    :return dict or list: API response with type conversion applied.
    """
    client = _get_binance_client()
    tuples = _dict_to_tuples(params)
    data = client.signed_request("GET", endpoint, tuples, recv_window=recv_window)
    return convert_response_type(data, decimal_mode=decimal_mode)


def signed_post(endpoint: str,
                decimal_mode: bool,
                params: dict | None = None,
                recv_window: int = 10000) -> dict | list:
    """
    Signed POST request via panzer BinanceClient.

    :param str endpoint: API endpoint path.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param dict params: Params for the request.
    :param int recv_window: Milliseconds of life for the request.
    :return dict or list: API response with type conversion applied.
    """
    client = _get_binance_client()
    tuples = _dict_to_tuples(params)
    data = client.signed_request("POST", endpoint, tuples, recv_window=recv_window)
    return convert_response_type(data, decimal_mode=decimal_mode)


def signed_delete(endpoint: str,
                  decimal_mode: bool,
                  params: dict | None = None,
                  recv_window: int = 10000) -> dict | list:
    """
    Signed DELETE request via panzer BinanceClient.

    :param str endpoint: API endpoint path.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param dict params: Params for the request.
    :param int recv_window: Milliseconds of life for the request.
    :return dict or list: API response with type conversion applied.
    """
    client = _get_binance_client()
    tuples = _dict_to_tuples(params)
    data = client.signed_request("DELETE", endpoint, tuples, recv_window=recv_window)
    quest_logger.info("signed_delete: endpoint=%s params=%s", endpoint, params)
    return convert_response_type(data, decimal_mode=decimal_mode)


def semi_signed_get(endpoint: str,
                    decimal_mode: bool,
                    params: dict | None = None) -> dict | list:
    """
    GET with API key header but no full signature (USER_STREAM / MARKET_DATA).

    :param str endpoint: API endpoint path.
    :param bool decimal_mode: Sets numeric data format to decimal.
    :param dict params: Params for the request.
    :return dict or list: API response with type conversion applied.
    """
    client = _get_binance_client()
    tuples = _dict_to_tuples(params)
    data = client.signed_request("GET", endpoint, tuples, sign=False)
    return convert_response_type(data, decimal_mode=decimal_mode)


def get_server_time() -> int:
    """
    Get time from server using panzer.

    :return int: A linux timestamp in milliseconds.
    """
    from .market import _get_panzer
    client = _get_panzer()
    resp = client.server_time()
    return resp['serverTime']
