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


def import_secret_module():
    """
    Imports the 'secret.py' module from anywhere in the project directory hierarchy.

    The function searches for the 'secret.py' module in the current directory and all upper directories until it finds it
    or reaches the root of the file system.

    :return: The 'secret' module if found, None otherwise.
    :rtype: Module or None
    """
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


class AesCipher(object):
    """
    Cyphering object.

    Initialization function. Generates a key and an initialization vector based on the CPU info and the user home.

    """

    def __init__(self):
        __seed = bytes(expanduser("~") + get_cpu_info()['brand_raw'], "utf8")
        self.__iv = hashlib.md5(__seed).hexdigest()
        self.__key = hashlib.md5(__seed[::-1]).hexdigest()

    def encrypt(self, msg: str) -> str:
        """
        Encrypting function. Encrypts a message with AES-128-CBC.

        :param str msg: Any message to encrypt.
        :return str: A bytes base64 string.
        """
        msg_padded = pad(msg.encode(), AES.block_size)
        cipher = AES.new(unhexlify(self.__key), AES.MODE_CBC, unhexlify(self.__iv))
        cipher_text = cipher.encrypt(msg_padded)
        return b64encode(cipher_text).decode('utf-8')

    def decrypt(self, msg_encrypted: str) -> str:
        """
        Decrypt function. Decrypts a message with AES-128-CBC.

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

    :return bool: True if supported, False otherwise.
    """
    min_version = (3, 7)
    max_version = (3, 11)  # Numba supports up to Python 3.10, so we set the maximum version to 3.11
    current_version = sys.version_info[:2]  # Get the first two elements of the version info tuple
    return min_version <= current_version <= max_version


def is_running_in_jupyter():
    """
    Check if the code is running in a Jupyter notebook.

    :return bool: True if running in Jupyter, False otherwise.
    """
    try:
        from ipykernel import connect
        return True
    except ImportError:
        return False
