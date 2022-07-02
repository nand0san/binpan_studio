from .logs import Logs
from .quest import get_signed_request, base_url, check_minute_weight
from time import time

wallet_logger = Logs(filename='./logs/wallet_logger.log', name='wallet_logger', info_level='INFO')


##########
# WALLET #
##########


def get_fees(symbol: str = None) -> dict:
    check_minute_weight(1)
    endpoint = '/sapi/v1/asset/tradeFee'
    timestamp = int(time() * 1000)  # ahorramos una llamada al servidor BNB
    if symbol:
        symbol = symbol.upper()
    ret = get_signed_request(base_url + endpoint, {'symbol': symbol, 'timestamp': timestamp})
    wallet_logger.debug(f"{ret}")
    return {i['symbol']: {'makerCommission': i['makerCommission'],
                          'takerCommission': i['takerCommission']} for i in ret}
