import requests
from base64 import b64encode, b64decode
import hashlib
from binascii import unhexlify
from os.path import expanduser
from cpuinfo import get_cpu_info
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class AesCipher(object):

    def __init__(self):
        __seed = bytes(expanduser("~") + get_cpu_info()['brand_raw'], "utf8")
        self.__iv = hashlib.md5(__seed).hexdigest()
        self.__key = hashlib.md5(__seed[::-1]).hexdigest()

    def encrypt(self, msg):
        msg_padded = pad(msg.encode(), AES.block_size)
        cipher = AES.new(unhexlify(self.__key), AES.MODE_CBC, unhexlify(self.__iv))
        cipher_text = cipher.encrypt(msg_padded)
        return b64encode(cipher_text).decode('utf-8')

    def decrypt(self, msg_encrypted):
        decipher = AES.new(unhexlify(self.__key), AES.MODE_CBC, unhexlify(self.__iv))
        plaintext = unpad(decipher.decrypt(b64decode(msg_encrypted)), AES.block_size).decode('utf-8')
        return plaintext


def get_exchange_limits() -> dict:
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
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/exchangeInfo'
    response = requests.get(base_url+endpoint).json()
    # info_dic = {k['symbol']: k for k in response['symbols']}

    limits = response['rateLimits']
    limits_dict = {}
    for limit in limits:
        if 'REQUEST' in limit['rateLimitType']:
            interval = str(limit['intervalNum'])
            interval += limit['interval'][0]
            limits_dict[f'REQUEST_{interval}'] = limit['limit']
        elif 'ORDERS' in limit['rateLimitType']:
            interval = str(limit['intervalNum'])
            interval += limit['interval'][0]
            limits_dict[f'ORDERS_{interval}'] = limit['limit']
        else:
            msg = f"BinPan error: Unknown limit from API: {limit}"
            raise Exception(msg)
    #
    #
    #
    # for i in limits:
    #     if i['rateLimitType'].upper() == 'ORDERS':
    #         k1 = f"x-mbx-order-count-{i['intervalNum']}{i['interval'][0].lower()}"
    #         k2 = f"x-mbx-order-count-{i['intervalNum']}{i['interval'][0].lower()}"
    #
    #     elif i['rateLimitType'].upper() == 'REQUEST_WEIGHT':
    #         k1 = f"x-mbx-used-weight-{i['intervalNum']}{i['interval'][0].lower()}"
    #         k2 = f"X-SAPI-USED-IP-WEIGHT-{i['intervalNum']}{i['interval'][0].upper()}"
    #
    #     elif i['rateLimitType'].upper() == 'RAW_REQUESTS':
    #         k1 = "x-mbx-used-weight"
    #         k2 = "x-mbx-used-weight"
    #     else:
    #         raise Exception("BinPan Rate Limit not parsed")
    #
    #     v = i['limit']
    #
    #     limits_dict[k1] = v
    #     limits_dict[k2] = v

    return limits_dict
