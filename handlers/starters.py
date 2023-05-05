"""
Functions to initialize before starting anything.
"""
import requests
from base64 import b64encode, b64decode
import hashlib
from binascii import unhexlify
from os.path import expanduser
from cpuinfo import get_cpu_info
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import sys
import os
import importlib
# import re


def import_secret_module():
    current_dir = os.path.abspath(os.curdir)

    while True:
        try:
            secret_module = importlib.import_module('secret')
            # regex = r"(?<=\\Users\\)([^\\]+)"
            # obfuscated_path = re.sub(regex, "XXXX", current_dir)
            # print("SECRET module found: ", obfuscated_path)
            return secret_module
        except ModuleNotFoundError:
            # Si no se encuentra el módulo, sube un nivel en el directorio
            parent_dir = os.path.dirname(current_dir)

            # Si ya estamos en la raíz del sistema de archivos, detener la búsqueda
            if parent_dir == current_dir:
                print("SECRET module not found!")
                return None

            current_dir = parent_dir
            sys.path.insert(0, current_dir)


def import_if_not_exists_(module_name: str):
    """
    Import a Python module if it hasn't been imported yet.

    This function takes a module_name parameter, which is the name of the module you want to import. If the module is not already present in sys.modules, it attempts to import the module using importlib.import_module().

    :param module_name: The name of the module to be imported.
    :type module_name: str

    :raises ImportError: If the specified module cannot be imported.

    :return: The imported module object if the module is successfully imported or if it is already present in sys.modules.
    :rtype: ModuleType

    :example:

    .. code-block:: python

        math_module = import_if_not_exists("math")
        print(math_module.sqrt(4))  # Prints "2.0"

    """
    if module_name not in sys.modules:
        try:
            importlib.import_module(module_name)
            print(f"Module {module_name} imported successfully.")
        except ImportError:
            raise Exception(f"Failed to import module {module_name}.")
    else:
        print(f"Module {module_name} already imported.")


class AesCipher(object):
    """
    Cyphering object.
    """

    def __init__(self):
        __seed = bytes(expanduser("~") + get_cpu_info()['brand_raw'], "utf8")
        self.__iv = hashlib.md5(__seed).hexdigest()
        self.__key = hashlib.md5(__seed[::-1]).hexdigest()

    def encrypt(self, msg: str) -> str:
        """
        Encrypting function.

        :param str msg: Any message to encrypt.
        :return str: A bytes base64 string.
        """
        msg_padded = pad(msg.encode(), AES.block_size)
        cipher = AES.new(unhexlify(self.__key), AES.MODE_CBC, unhexlify(self.__iv))
        cipher_text = cipher.encrypt(msg_padded)
        return b64encode(cipher_text).decode('utf-8')

    def decrypt(self, msg_encrypted: str) -> str:
        """
        Decrypt function.

        :param str msg_encrypted: A bytes base64 string.
        :return str: Plain text.
        """
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
    response = requests.get(base_url + endpoint).json()
    # info_dic = {k['symbol']: k for k in response['symbols']}
    try:
        limits = response['rateLimits']
    except KeyError:
        print(response)
        print(response.keys())
        limits = [{'rateLimitType': 'REQUEST_WEIGHT', 'interval': 'MINUTE', 'intervalNum': 1, 'limit': 1200},
                  {'rateLimitType': 'ORDERS', 'interval': 'SECOND', 'intervalNum': 10, 'limit': 50},
                  {'rateLimitType': 'ORDERS', 'interval': 'DAY', 'intervalNum': 1, 'limit': 160000},
                  {'rateLimitType': 'RAW_REQUESTS', 'interval': 'MINUTE', 'intervalNum': 5, 'limit': 6100}]

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

    return limits_dict


def is_python_version_numba_supported() -> bool:
    """
    Verify if python version is numba supported.
    """
    min_version = (3, 7)
    max_version = (3, 10)
    current_version = sys.version_info
    return min_version <= current_version <= max_version
