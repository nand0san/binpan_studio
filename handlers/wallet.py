import pandas as pd

from .logs import Logs
from .quest import api_raw_signed_get
from .time_helper import convert_milliseconds_to_str
from .time_helper import convert_string_to_milliseconds

wallet_logger = Logs(filename='./logs/wallet_logger.log', name='wallet_logger', info_level='INFO')


##########
# WALLET #
##########


def convert_str_date_to_ms(date: str or int,
                           time_zone: str):
    """
    Converts dates strings formatted as "2022-05-11 06:45:42" to timestamp in milliseconds.

    :param str or int date: Date to check format.
    :param time_zone: A time zone like 'Europe/Madrid'
    :return int: Milliseconds of timestamp.
    """
    if type(date) == str:
        date = convert_string_to_milliseconds(date, timezoned=time_zone)
    return date


def daily_account_snapshot(account_type: str = 'SPOT',
                           startTime: int or str = None,
                           endTime: int = None,
                           limit=30,
                           time_zone=None) -> pd.DataFrame:
    """
    The query time period must be inside the previous 30 days.
    Support query within the last month, one month only.
    If startTime and endTime not sent, return records of the last days by default.

    Weight(IP): 2400

    :param str account_type: SPOT or MARGIN
    :param int limit: Days limit. Default 30.
    :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
    :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
    :param str time_zone: A time zone to parse index like 'Europe/Madrid'
    :return pd.DataFrame:

    """
    startTime = convert_str_date_to_ms(date=startTime, time_zone=time_zone)
    endTime = convert_str_date_to_ms(date=endTime, time_zone=time_zone)

    available = ['SPOT', 'MARGIN']
    if account_type not in available:
        exc = f"BinPan error: {account_type} not in {available}"
        wallet_logger.error(exc)
        raise Exception(exc)
    elif limit > 30 or limit < 7:
        exc = f"BinPan error: {limit} is over the maximum 30 or under the minimum 7."
        wallet_logger.error(exc)
        raise Exception(exc)

    ret = api_raw_signed_get(endpoint='/sapi/v1/accountSnapshot',
                             params={'type': account_type,
                                     'startTime': startTime,
                                     'endTime': endTime,
                                     'limit': limit},
                             weight=2400)
    rows = []
    for update in ret['snapshotVos']:
        updateTime = update['updateTime']
        data = update['data']
        totalAssetOfBtc = float(data['totalAssetOfBtc'])
        if account_type == 'MARGIN':
            assets_field = 'userAssets'
        else:
            assets_field = 'balances'
        for balance in data[assets_field]:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                row = {'updateTime': updateTime,
                       'totalAssetOfBtc': totalAssetOfBtc,
                       'asset': balance['asset'],
                       'free': free,
                       'locked': locked}
                if time_zone:
                    row['datetime'] = convert_milliseconds_to_str(ms=updateTime, timezoned=time_zone)
                rows.append(row)

    if time_zone:
        ret = pd.DataFrame(rows)
        if not ret.empty:
            ret = ret.set_index('datetime').sort_index()
            ret.index.name = f"{account_type} {time_zone}"
    else:
        ret = pd.DataFrame(rows)
        if not ret.empty:
            ret = ret.set_index('updateTime').sort_index()
            ret.index.name = f"{account_type} timestamp"
    return ret


##########
# trades #
##########


def get_spot_trades_list(symbol: str,
                         limit: int = 1000,
                         recvWindow: int = 10000) -> list:
    endpoint = '/api/v3/myTrades'
    return api_raw_signed_get(endpoint=endpoint,
                              params={'limit': limit,
                                      'symbol': symbol,
                                      'recvWindow': recvWindow},
                              weight=10)


############
# balances #
############


def get_spot_account_info(recvWindow: int = 10000) -> dict:
    endpoint = '/api/v3/account'
    return api_raw_signed_get(endpoint=endpoint,
                              params={'recvWindow': recvWindow},
                              weight=10)


def spot_free_balances_parsed(data_dic: dict = None) -> dict:
    ret = {}
    if not data_dic:
        data_dic = get_spot_account_info()
    wallet_logger.debug(f"spot_free_balances_parsed len(get_spot_account_info()):{len(data_dic)}")
    for asset in data_dic['balances']:
        qty = float(asset['free'])
        if qty:
            ret.update({asset['asset']: qty})
    return ret


def spot_locked_balances_parsed(data_dic: dict = None) -> dict:
    ret = {}
    if not data_dic:
        data_dic = get_spot_account_info()
    wallet_logger.debug(f"spot_locked_balances_parsed get_spot_account_info: {len(data_dic)}")

    for asset in data_dic['balances']:
        qty = float(asset['locked'])
        if qty:
            ret.update({asset['asset']: qty})
    return ret


def get_coins_with_balance() -> list:
    """Retorna una lista de símbolos con balance positivo en la cuenta"""
    data_dic = get_spot_account_info()
    free = spot_free_balances_parsed(data_dic=data_dic)
    locked = spot_locked_balances_parsed(data_dic=data_dic)
    free = [symbol for symbol, balance in free.items() if float(balance) > 0]
    locked = [symbol for symbol, balance in locked.items() if float(balance) > 0]
    symbols = free + locked
    return list(set(symbols))
