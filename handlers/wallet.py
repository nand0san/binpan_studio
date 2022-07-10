import pandas as pd

from .logs import Logs
from .quest import api_raw_signed_get
from .time_helper import convert_milliseconds_to_str

wallet_logger = Logs(filename='./logs/wallet_logger.log', name='wallet_logger', info_level='INFO')


##########
# WALLET #
##########


# def get_fees(symbol: str = None) -> dict:
#     """
#     Returns fees for a symbol or for every symbol if not passed.
#     :param symbol:
#     :return:
#     """
#     check_minute_weight(1)
#     endpoint = '/sapi/v1/asset/tradeFee'
#     timestamp = int(time() * 1000)  # ahorramos una llamada al servidor BNB
#     if symbol:
#         symbol = symbol.upper()
#     ret = get_signed_request(base_url + endpoint, {'symbol': symbol, 'timestamp': timestamp})
#     wallet_logger.debug(f"{ret}")
#     return {i['symbol']: {'makerCommission': i['makerCommission'],
#                           'takerCommission': i['takerCommission']} for i in ret}

def daily_account_snapshot(account_type: str = 'SPOT',
                           startTime: int = None,
                           endTime: int = None,
                           limit=30,
                           time_zone=None) -> pd.DataFrame:
    """
    The query time period must be less than 30 days.
    Support query within the last month, one month only.
    If startTime and endTime not sent, return records of the last 7 days by default.

    Weight(IP): 2400

    :param account_type:
    :param limit: Days limit.
    :param int startTime: Period bounding for the snapshot.
    :param int endTime: Period bounding for the snapshot.
    :param str time_zone: A time zone to parse index like 'Europe/Madrid'
    :return pd.DataFrame:

    """
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
